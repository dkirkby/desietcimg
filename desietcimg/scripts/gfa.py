"""Scripts to process GFA data.
"""
import os
import sys
import argparse
import warnings

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from desietcimg.gfa import *
from desietcimg.plot import *
from desietcimg.util import *


def process(night, expid, args):
    """Process a single exposure.
    """
    night = str(night)
    expid = '{0:08d}'.format(int(expid))
    logging.info('Processing {0}/{1}'.format(night, expid))


def gfadiq():
    parser = argparse.ArgumentParser(
        description='Summarize delivered image quality for a GFA exposure.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--night', type=int, metavar='YYYYMMDD',
        help='Night of exposure to process in the format YYYYMMDD')
    parser.add_argument('--expid', type=int, metavar='N',
        help='Exposure sequence identifier to process')
    parser.add_argument('--inpath', type=str, metavar='PATH',
        help='Path where raw data is organized under YYYYMMDD directories')
    parser.add_argument('--outpath', type=str, metavar='PATH',
        help='Path where outputs willl be organized under YYYYMMDD directories')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='Path where logging output should be written')
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(filename=args.logpath, level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # Are we running on a recognized host with special defaults?
    host = None
    if os.getenv('NERSC_HOST') is not None:
        host = 'NERSC'
        logging.info('Detected a NERSC host.')

    # Determine where the input raw data is located.
    if args.inpath is None:
        if host is 'NERSC':
            inpath = '/project/projectdirs/desi/spectro/data/'
        else:
            print('No input path specified with --inpath.')
            sys.exit(-1)
    args.inpath = Path(args.inpath)
    if not args.inpath.exists():
        print('Non-existant input path: {0}'.format(args.inpath))
        sys.exit(-2)
    logging.info('Input path is {0}'.format(args.inpath))

    # Determine where the outputs will go.
    if args.outpath is None:
        print('No output path specified with --outpath.')
        sys.exit(-1)
    args.outpath = Path(args.outpath)
    if not args.outpath.exists():
        print('Non-existant output path: {0}'.format(args.outpath))
        sys.exit(-2)
    logging.info('Output path is {0}'.format(args.outpath))

    process(args.night, args.expid, args)
