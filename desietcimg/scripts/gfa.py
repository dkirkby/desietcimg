"""Scripts to process GFA data.
"""
import os
import sys
import argparse
import warnings
import glob
import multiprocessing

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


def process_one(inpath, camera, framepath):
    """Process a single camera of a single exposure.
    """
    global GFA
    with fitsio.FITS(str(inpath), mode='r') as hdus:
        if camera not in hdus:
            logging.error('Missing HDU {0}'.format(camera))
            return None
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


def process(night, expid, args, pool, pool_timeout=5):
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
    # Are there already existing outputs?
    fitspath = outpath / 'stack_{0}.fits'.format(expid)
    if fitspath.exists() and not args.overwrite:
        logging.info('Will not overwrite outputs in {0}'.format(outpath))
        return
    # Open the FITS file to read.
    logging.info('Processing {0}'.format(inpath))
    # Process each camera in the input.
    results = {}
    framepath = outpath if args.save_frames else None
    for camera in GFA.gfa_names:
        if pool is None:
            result = process_one(inpath, camera, framepath)
            if result is None:
                logging.error('Error processing HDU {0}'.format(camera))
            else:
                results[camera], meta = result
        else:
            results[camera] = pool.apply_async(process_one, (inpath, camera, framepath))
    if pool:
        # Collect the pooled results.
        for camera in results:
            try:
                result = results[camera].get(timeout=pool_timeout)
                if result is None:
                    logging.error('Error processing HDU {0}'.format(camera))
                else:
                    results[camera], meta = result
            except TimeoutError:
                logging.error('Timeout waiting for {0} pool result'.format(camera))
                del results[camera]
    # Save the output FITS file.
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
    parser.add_argument('--batch', action='store_true',
        help='Process all existing exposures on night')
    parser.add_argument('--watch', action='store_true',
        help='Wait for and process new exposures on night')
    parser.add_argument('--save-frames', action='store_true',
        help='Save images of each GFA frame')
    parser.add_argument('--overwrite', action='store_true',
        help='Overwrite existing outputs')
    parser.add_argument('--inpath', type=str, metavar='PATH',
        help='Path where raw data is organized under YYYYMMDD directories')
    parser.add_argument('--outpath', type=str, metavar='PATH',
        help='Path where outputs willl be organized under YYYYMMDD directories')
    parser.add_argument('--calibpath', type=str, metavar='PATH',
        help='Path to GFA calibration FITS file to use')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='Path where logging output should be written')
    parser.add_argument('--npool', type=int, default=0, metavar='N',
        help='Number of workers in multiprocessing pool')
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

    if args.night is None:
        print('Missing required argument: night.')
        sys.exit(-1)
    nightpath = args.inpath / str(args.night)
    if not nightpath.exists():
        print('Non-existant directory for night: {0}'.format(nightpath))

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

    if args.npool > 0:
        # Initialize multiprocessing.
        context = multiprocessing.get_context(method='fork')
        if args.npool > context.cpu_count():
            args.npool = context.cpu_count()
            log.warning('Reducing --npool to {0}'.format(args.npool))
        pool = context.Pool(processes=args.npool)
        logging.info('Initialized multiprocessing pool with {0} workers'.format(args.npool))
    else:
        pool = None

    if args.expid is not None:
        process(args.night, args.expid, args, pool)
        return

    if args.batch or args.watch:
        current_exposures = lambda: set((int(str(path)[-8:]) for path in nightpath.glob('????????')))
        # Find the existing exposures on this night.
        existing = current_exposures()
        if args.batch:
            for expid in existing:
                process(args.night, expid, args, pool)