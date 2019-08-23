"""Perform calibration analysis.
"""
import argparse
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from desietcimg.calib import *
from desietcimg.plot import *
from desietcimg.db import *


def etccalib():
    parser = argparse.ArgumentParser(
        description='Analyze calibration data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--ci-night', type=int, default=0, metavar='YYYYMMDD',
        help='night of CI run to analyze in the format YYYYMMDD')
    parser.add_argument('--saveimg', action='store_true',
        help='save images showing calibration results')
    parser.add_argument('--outpath', type=str, default='.',
        help='path where output files are saved')
    parser.add_argument('--db', type=str, default='db.yaml',
        help='yaml file of database connection parameters')
    args = parser.parse_args()

    if args.ci_night > 0:
        # Initialize the online database.
        db = DB(args.db)
        # Find all zero and dark exposures on this night.
        zero_exps = db.select(
            'exposure.exposure', 'id,night,exptime', limit=1000,
            where="sequence='CI' and flavor='zero' and night={0}".format(args.ci_night))
        dark_exps = db.select(
            'exposure.exposure', 'id,night,exptime', limit=1000,
            where="sequence='CI' and flavor='dark' and night={0}".format(args.ci_night))
        if args.verbose:
            print('Found {0} zero and {1} dark exposures from {2}.'
                  .format(len(zero_exps), len(dark_exps), args.ci_night))
        # Process each camera separately. This is somewhat slower but uses 1/5 the memory.
        cameras = 'CIN', 'CIE', 'CIS', 'CIW', 'CIC'
        for camera in cameras:
            CA = CalibrationAnalysis(camera, 2048, 3072)
