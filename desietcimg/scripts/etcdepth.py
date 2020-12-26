import argparse
import logging
import pdb
import traceback
import sys
from pathlib import Path

import desietcimg.db


def etcdepth(args):
    # Check required paths.
    etcpath = Path(args.etcpath)
    if not etcpath.exists():
        raise RuntimeError(f'Non-existent --etcpath {etcpath}')
    # Initialize online database access.
    db = desietcimg.db.DB(http_fallback=not args.direct)
    # Connect to the exposures table.
    expdb = desietcimg.db.Exposures(db, 'id,night,tileid,exptime,mjd_obs,program')
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
            etc = etcpath / night / exptag
            if not etc.exists():
                logging.error(f'Missing ETC exposure data for {night}/{exptag}')
            sky = etc / f'sky_{exptag}.csv'
            if not sky.exists():
                logging.warning(f'Missing ETC sky data for {night}/{exptag}')


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
