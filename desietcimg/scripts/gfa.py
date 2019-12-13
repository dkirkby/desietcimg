"""Scripts to process GFA data.
"""
import os
import sys
import argparse
import warnings

from pathlib import Path

import numpy as np

try:
    from PIL import Image
    img_format = 'jpg'
except ImportError:
    img_format = 'png'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fitsio

from desietcimg.gfa import *
from desietcimg.plot import *
from desietcimg.util import *

# Globals shared by process_one below.
GFA = None
raw = np.empty((1, 1032, 2248), np.uint32)


def process_one(hdus, camera, framepath):
    """Process a single camera of a single exposure.
    """
    hdr = hdus[camera].read_header()
    meta = {}
    for key in 'NIGHT', 'EXPID', 'MJD-OBS', 'EXPTIME', 'GCCDTEMP':
        if key not in hdr:
            logging.error('Missing required key {0} for {1}'.format(key, camera))
            return None
        meta[key] = hdr[key]
    raw[0] = hdus[camera][:, :]
    logging.info('Processing {0}'.format(camera))
    try:
        GFA.setraw(raw, name=camera)
    except ValueError as e:
        logging.error(e)
        return None
    GFA.data -= GFA.get_dark_current(meta['GCCDTEMP'], meta['EXPTIME'])
    if camera.startswith('GUIDE'):
        GFA.get_psfs()
        stamps = GFA.psfs
        result = GFA.psf_stack
    else:
        GFA.get_donuts()
        stamps = GFA.donuts[0] + GFA.donuts[1]
        result = GFA.donut_stack
    if framepath is not None:
        label = '{0} {1} {2:.1f}s'.format(meta['NIGHT'], meta['EXPID'], meta['EXPTIME'])
        plot_data(GFA.data[0], GFA.ivar[0], downsampling=2, label=label, stamps=stamps, colorhist=True)
        plt.savefig(framepath / '{0}_{1:08d}.{2}'.format(camera, meta['EXPID'], img_format), quality=80)
        plt.clf()
    return result, meta


def process(night, expid, args):
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
    # Open the FITS file to read.
    logging.info('Processing {0}'.format(inpath))
    hdus = fitsio.FITS(str(inpath), mode='r')
    # Process each camera in the input.
    results = {}
    for camera in GFA.gfa_names:
        if camera not in hdus:
            logging.error('Missing HDU {0}'.format(camera))
            continue
        result, meta = process_one(hdus, camera, outpath if args.save_frames else None)
        if result is None:
            logging.error('Error processing HDU {0}'.format(camera))
        else:
            results[camera] = result
    # Save the output FITS file.
    fitspath = outpath / 'stack_{0}.fits'.format(expid)
    with fitsio.FITS(str(fitspath), 'rw', clobber=True) as hdus:
        for camera in results:
            if camera.startswith('GUIDE'):
                hdus.write(np.stack(results[camera]).astype(np.float32), extname=camera)
            else:
                L, R = results[camera]
                if L is not None:
                    hdus.write(np.stack(L).astype(np.float32), extname=camera + 'L')
                if R is not None:
                    hdus.write(np.stack(R).astype(np.float32), extname=camera + 'R')
    # Produce a summary plot.
    fig = plot_image_quality(results, meta)
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
    parser.add_argument('--save-frames', action='store_true',
        help='Save images of each GFA frame')
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
    elif os.getenv('DOS_HOME') is not None:
        host = 'DOS'

    # Determine where the input raw data is located.
    if args.inpath is None:
        if host is 'NERSC':
            args.inpath = '/project/projectdirs/desi/spectro/data/'
        elif host is 'DOS':
            args.inpath = '/exposures/desi/'
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
        elif host is 'DOS':
            # Should use a more permanent path than this which is synched via svn.
            args.calibpath = '/n/home/desiobserver/donut/GFA_calib.fits'
        else:
            print('No GFA calibration data path specified with --calibpath.')
            sys.exit(-1)
    args.calibpath = Path(args.calibpath)
    if not args.calibpath.exists():
        print('Non-existant GFA calibration path: {0}'.format(args.calibpath))
        sys.exit(-2)

    # Initialize the GFA analysis object.
    global GFA
    GFA = GFACamera(calib_name=args.calibpath)

    if args.night is None or args.expid is None:
        print('Must specify night and expid for now.')
        sys.exit(-1)

    process(args.night, args.expid, args)
