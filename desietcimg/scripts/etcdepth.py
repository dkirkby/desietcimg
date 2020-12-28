import argparse
import logging
import pdb
import traceback
import sys
import os
from pathlib import Path

import numpy as np

import fitsio

import desietcimg.db
import desietcimg.spectro


# TODO: propagate actual GFA, SKY exposure times from FITS file


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
        # Calculate ivar-weighted mean relative flux.
        flux[frame] = np.sum(ivar * data['flux'][sel]) / np.sum(ivar) / exptime
    return mjd, flux


def load_etc_gfa(names, exptime):
    mjd, transp, ffrac = [], [], []
    for name in names:
        data = np.loadtxt(name, delimiter=',', dtype=
            {'names':('mjd','dx','dy','transp','ffrac','nll'),
             'formats': ('f8','f4','f4','f4','f4','f4')})
        # Calculate the median over GFA cameras in each frame.
        mjd.append(np.mean(data['mjd'], axis=0) + 0.5 * exptime / 86400)
        transp.append(np.nanmedian(data['transp'], axis=0))
        ffrac.append(np.nanmedian(data['ffrac'], axis=0))
    return np.array(mjd), np.array(transp), np.array(ffrac)


def load_spec_sky(names, exptime):
    hdr_exptime = None
    detected = {camera: desietcimg.spectro.CoAdd(camera) for camera in 'brz'}
    for name in names:
        camera = name.name[4]
        assert camera in 'brz'
        spec = int(name.name[5])
        assert spec in range(10)
        # Read the flat-fielded sky model in this (spectrograph, camera)
        with fitsio.FITS(str(name)) as hdus:
            if hdr_exptime is None:
                hdr_exptime = hdus[0].read_header()['EXPTIME']
                if abs(hdr_exptime - exptime) > 1:
                    logging.warning(f'Actual exptime {hdr_exptime:.1f}s != requested {exptime:.1f}s.')
            else:
                if hdus[0].read_header()['EXPTIME'] != hdr_exptime:
                    logging.error(f'Inconsistent header EXPTIME for {name}.')
            flux = hdus['SKY'].read()
            ivar = hdus['IVAR'].read()
            if np.any(ivar == 0):
                logging.warning(f'Found {np.count_nonzero(ivar == 0)} / {ivar.size} zero ivar pixels in {name}.')
            #mask = hdus['MASK'].read()
            # Verify that all masked pixels have zero ivar.
            #assert np.all(ivar[mask != 0] == 0)


def etcdepth(args):
    # Check required paths.
    DESIROOT = Path(args.desiroot or os.getenv('DESI_ROOT', '/global/cfs/cdirs/desi'))
    logging.info(f'DESIROOT={DESIROOT}')
    SPEC = DESIROOT / 'spectro' / 'redux' / args.release / 'exposures'
    if not SPEC.exists():
        raise RuntimeError(f'Non-existent {SPEC}')
    logging.info(f'SPEC={SPEC}')
    ETC = Path(args.etcpath or (DESIROOT / 'spectro' / 'ETC'))
    if not ETC.exists():
        raise RuntimeError(f'Non-existent {ETC}')
    logging.info(f'ETC={ETC}')
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
        expdata = expdb.select(
            f"tileid IN ({args.tiles}) AND exptime>={args.min_exptime} AND night>20200100 AND flavor='science'",
            maxrows=1000)
    elif args.tiles == 'SV1':
        expdata = expdb.select(
            db.where(night=(20201201,None), exptime=(args.min_exptime,None), program='SV%', flavor='science'),
            maxrows=1000)
    else:
        raise ValueError(f'Cannot interpret --tiles {args.tiles}')
    # Loop over exposures for each tile.
    nexp = len(expdata)
    tiles = set(expdata['tileid'])
    logging.info(f'Processing {nexp} exposures for tiles: {tiles}')
    for tile in tiles:
        sel = expdata['tileid'] == tile
        logging.info(f'tile {tile} exposures {list(expdata["id"][sel])}')
        for _, row in expdata[sel].iterrows():
            night = str(int(row['night']))
            expid = int(row['id'])
            exptag = f'{expid:08d}'
            mjd_spectro = row['mjd_obs']
            exptime_spectro = row['exptime']
            etcdir = ETC / night / exptag
            '''
            if not etcdir.exists():
                logging.error(f'Missing ETC exposure data for {night}/{exptag}')
            else:
                # Process ETC data for this exposure.
                exptime_sky = row['skytime']
                exptime_gfa = row['guidtime']
                if exptime_gfa is None:
                    #logging.warning(f'Forcing GUIDTIME=5 for {night}/{exptag}')
                    exptime_gfa = 5
                sky = etcdir / f'sky_{exptag}.csv'
                if sky.exists():
                    mjd_sky, flux_sky = load_etc_sky(sky, exptime_sky)
                else:
                    logging.warning(f'Missing ETC sky data for {night}/{exptag}')
                gfas = sorted(etcdir.glob(f'guide_GUIDE?_{exptag}.csv'))
                if gfas:
                    mjd_gfa, transp_gfa, ffrac_gfa = load_etc_gfa(gfas, exptime_gfa)
                else:
                    logging.warning(f'Missing ETC guide data for {night}/{exptag}')
            '''
            specdir = SPEC / night / exptag
            if not specdir.exists():
                logging.error(f'Missing SPEC exposure data for {night}/{exptag}')
            else:
                # Process SPEC data for this exposure.
                skys = sorted(specdir.glob(f'sky-??-{exptag}.fits'))
                if skys:
                    #load_spec_sky(skys, exptime_spectro)
                    pass
                else:
                    logging.warning(f'Missing SPEC sky data for {night}/{exptag}.')


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
    parser.add_argument('--release', type=str, default='daily',
        help='pipeline reduction release to use')
    parser.add_argument('--desiroot', type=str, default=None,
        help='root path for locating DESI data, defaults to $DESI_ROOT')
    parser.add_argument('--etcpath', type=str, default=None,
        help='path where ETC outputs are stored, defaults to <desiroot>/ETC')
    parser.add_argument('--min-exptime', type=float, default=100,
        help='ignore exposures of duration less than this value')
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
