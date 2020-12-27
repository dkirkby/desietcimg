import argparse
import logging
import pdb
import traceback
import sys
from pathlib import Path

import numpy as np

import desietcimg.db


def load_etc_sky(name, exptime):
    """Read the ETC results for a sky camera exposure from a CSV file.
    Return arrays of MJD and relative flux values.
    """
    data = np.loadtxt(
        name, delimiter=',', dtype=
        {'names':('camera','frame','mjd','flux','dflux','chisq','nfiber'),
            'formats':('S7','i4','f8','f4','f4','f4','i4')})
    # Calculate a weighted average of sky camera flux in each frame.
    frames = np.unique(data['frame'])
    nframe = len(frames)
    if not np.all(frames == np.arange(nframe)):
        logging.warning(f'Unexpected sky frames in {name}')
        return None, None
    mjd = np.empty(nframe)
    flux = np.empty(nframe)
    for frame in frames:
        sel = data['frame'] == frame
        # Calculate sky exposure midpoint.
        mjd[frame] = np.mean(data['mjd'][sel]) + 0.5 * exptime / 86400
        ivar = data['dflux'][sel] ** -0.5
        # Calculate ivar-weighted mean flux.
        flux[frame] = np.sum(ivar * data['flux'][sel]) / np.sum(ivar)
    return mjd, flux


def load_etc_gfa(names, exptime):
    mjd, transp, ffrac = [], [], []
    for name in names:
        data = np.loadtxt(name, delimiter=',', dtype=
            {'names':('mjd','dx','dy','transp','ffrac','nll'),
             'formats': ('f8','f4','f4','f4','f4','f4')})
        # Calculate the median over GFA cameras in each frame.
        mjd.append(np.mean(data['mjd'], axis=0))
        transp.append(np.nanmedian(data['transp'], axis=0))
        ffrac.append(np.nanmedian(data['ffrac'], axis=0))
    return np.array(mjd), np.array(transp), np.array(ffrac)


def etcdepth(args):
    # Check required paths.
    etcpath = Path(args.etcpath)
    if not etcpath.exists():
        raise RuntimeError(f'Non-existent --etcpath {etcpath}')
    # Initialize online database access.
    db = desietcimg.db.DB(http_fallback=not args.direct)
    # Connect to the exposures table.
    expdb = desietcimg.db.Exposures(db, 'id,night,tileid,exptime,skytime,guidtime,mjd_obs,program')
    # Determine the list of tiles to process.
    tiles = set(args.tiles.split(','))
    try:
        numeric = all([int(tile) > 0 for tile in tiles])
    except ValueError:
        numeric = False
    if numeric:
        expdata = expdb.select(f'tileid IN ({args.tiles})', maxrows=1000)
    elif args.tiles == 'SV1':
        expdata = expdb.select(
            db.where(night=(20201201,None), exptime=(45,None), program='SV%'), maxrows=1000)
    else:
        raise ValueError(f'Cannot interpret --tiles {args.tiles}')
    # Loop over exposures for each tile.
    nexp = len(expdata)
    tiles = set(expdata['tileid'])
    logging.info(f'Processing {nexp} exposures for tiles: {tiles}')
    for tile in tiles:
        print('tile', tile)
        sel = expdata['tileid'] == tile
        for _, row in expdata[sel].iterrows():
            night = str(row['night'])
            expid = row['id']
            exptag = f'{expid:08d}'
            mjd_spectro = row['mjd_obs']
            exptime_spectro = row['exptime']
            exptime_sky = row['skytime']
            exptime_gfa = row['guidtime']
            etc = etcpath / night / exptag
            if not etc.exists():
                logging.error(f'Missing ETC exposure data for {night}/{exptag}')
            sky = etc / f'sky_{exptag}.csv'
            if sky.exists():
                mjd_sky, flux_sky = load_etc_sky(sky, exptime_sky)
            else:
                logging.warning(f'Missing ETC sky data for {night}/{exptag}')
                mjd_sky, flux_sky = None, None
            gfas = sorted(etc.glob(f'guide_GUIDE?_{exptag}.csv'))
            if gfas:
                mjd_gfa, transp_gfa, ffrac_gfa = load_etc_gfa(gfas, exptime_gfa)
            else:
                logging.warning(f'Missing ETC guide data for {night}/{exptag}')
                mjd_gfa, transp_gfa, ffrac_gfa = None, None, None


def main():
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(
        description='Calculate per-exposure effective depths',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='path where logging output should be written')
    parser.add_argument('--debug', action='store_true',
        help='print traceback and enter debugger after an exception')
    parser.add_argument('--tiles', type=str,
        help='comma-separated list of tiles or a predefined name like SV1')
    parser.add_argument('--etcpath', type=str, required=True,
        help='path where ETC outputs are stored')
    parser.add_argument('--direct', action='store_true',
        help='database connection must be direct')
    args = parser.parse_args()

    # Configure logging.
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(filename=args.logpath, level=level,
        format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y%m%d %H:%M:%S')

    try:
        retval = etcdepth(args)
        sys.exit(retval)
    except Exception as e:
        if args.debug:
            # https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            print(e)
            sys.exit(-1)
