"""Perform guide camera analysis on commissioning instrument data.
"""
import argparse
from pathlib import Path

import numpy as np

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
    parser.add_argument('--db', type=str, default='db.yaml',
        help='yaml file of database connection parameters')
    args = parser.parse_args()

    db = DB(args.db)
    ExpInfo = Exposures(db)

    GCA = GuideCameraAnalysis()
