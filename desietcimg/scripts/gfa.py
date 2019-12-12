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

import fitsio

from desietcimg.gfa import *
from desietcimg.plot import *
from desietcimg.util import *


def process(GFA, night, expid, args):
    """Process a single exposure.
    """
    # Locate the exposure path.
    night = str(night)
    expid = '{0:08d}'.format(int(expid))
    inpath = args.inpath / night / expid
    if not inpath.exists():
        logging.error('Non-existant path: {0}'.format(inpath))
        return
    # Look for a GFA SCIENCE exposure.
    inpath = inpath / 'gfa-{0}.fits.fz'.format(expid)
    if not inpath.exists():
        logging.error('Non-existant file: {0}'.format(inpath))
        return
    # Prepare the output path.
    outpath = args.outpath / night / expid
    outpath.mkdir(parents=True, exist_ok=True)
    # Process this exposure to produce a stack FITS file.
    logging.info('Processing {0}'.format(inpath))
    fitspath = outpath / 'stack_{0}.fits'.format(expid)
    GFA.process(inpath, fitspath)
    # Read the generated FITS file.
    stacks = {}
    with fitsio.FITS(str(fitspath)) as hdus:
        header = hdus[0].read_header()
        meta = {k: header[k] for k in ('NIGHT', 'EXPID', 'EXPTIME', 'MJD-OBS')}
        for hdu in hdus[1:]:
            stacks[hdu.get_extname()] = hdu.read().copy()
    # Produce a summary plot.
    fig = plot_image_quality(stacks, meta)
    # Save the summary plot.
    plt.savefig(str(outpath / 'gfadiq_{0}.png'.format(expid)))
    plt.clf()


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
    parser.add_argument('--calibpath', type=str, metavar='PATH',
        help='Path to GFA calibration FITS file to use')
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
            args.inpath = '/project/projectdirs/desi/spectro/data/'
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

    # Locate the GFA calibration data.
    if args.calibpath is None:
        if host is 'NERSC':
            args.calibpath = '/global/project/projectdirs/desi/cmx/gfa/calib/GFA_calib.fits'
        else:
            print('No GFA calibration data path specified with --calibpath.')
            sys.exit(-1)
    args.calibpath = Path(args.calibpath)
    if not args.calibpath.exists():
        print('Non-existant GFA calibration path: {0}'.format(args.calibpath))
        sys.exit(-2)

    # Initialize the GFA analysis object.
    GFA = GFACamera(calib_name=args.calibpath)

    if args.night is None or args.expid is None:
        print('Must specify night and expid for now.')
        sys.exit(-1)

    process(GFA, args.night, args.expid, args)
