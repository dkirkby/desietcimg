"""Perform guide camera analysis on commissioning instrument data.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

import fitsio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from desietcimg.guide import *
from desietcimg.plot import *
from desietcimg.db import *


def ciproc():
    parser = argparse.ArgumentParser(
        description='Analyze CI data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--stamp-size', type=int, default=65,
        help='stamp size to use for analysis, must be odd')
    parser.add_argument('--nsrc', type=int, default=12,
        help='number of candiate PSF sources to detect')
    parser.add_argument('--db', type=str, default='db.yaml',
        help='yaml file of database connection parameters')
    parser.add_argument('input', nargs='+', type=str,
        help='FITS file, sequence number, or file containing a list of sequence numbers')
    args = parser.parse_args()

    GCA = GuideCameraAnalysis(stamp_size=args.stamp_size)
    ExpInfo = None

    warnings.simplefilter(action='ignore', category=FutureWarning)

    for input in args.input:
        try:
            seqnum = int(input)
            print('seqnum', seqnum)
            continue
        except ValueError:
            pass
        if not os.path.exists(input):
            print('No such file: {0}.'.format(input))
            continue
        with fitsio.FITS(input, 'r') as hdus:
            meta = hdus[0].read_header()
            for camera in 'CIN', 'CIE', 'CIS', 'CIW', 'CIC':
                if camera in hdus:
                    D = hdus[camera][0, :, :][0]
                    GCR = GCA.detect_sources(D, meta=meta, nsrc=args.nsrc)
                    if args.verbose:
                        print('== {0}:'.format(camera))
                        GCR.print()

    #db = DB(args.db)
    #ExpInfo = Exposures(db)
