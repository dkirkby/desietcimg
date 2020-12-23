import argparse
import logging
import pdb
import traceback
import sys

import desietcimg.db


def etcdepth(args):
    # Initialize online database access.
    db = desietcimg.db.DB(http_fallback=not args.direct)
    # Connect to the exposures table.
    exposures = desietcimg.db.Exposures(db, 'id,night,exptime,mjd_obs')
    # Do a small test request.
    info = exposures(69409)
    print(info)


def main():
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(
        description='Calculate per-exposure effective depths',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='Path where logging output should be written')
    parser.add_argument('--debug', action='store_true',
        help='Print traceback and enter debugger after an exception')
    parser.add_argument('--direct', action='store_true',
        help='Database connection must be direct')
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
