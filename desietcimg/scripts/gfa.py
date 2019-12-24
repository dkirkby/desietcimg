"""Scripts to process GFA data.
"""
import os
import sys
import time
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


def process_one(inpath, night, expid, guiding, camera, exptime, ccdtemp, framepath):
    """Process a single camera of a single exposure.
    """
    global GFA
    with fitsio.FITS(str(inpath), mode='r') as hdus:
        if camera not in hdus:
            logging.error('Missing HDU {0}'.format(camera))
            return None
        logging.info('Processing {0}'.format(camera))
        if guiding:
            raw[0] = hdus[camera][0, :, :]
            exptime, ccdtemp = exptime[0], ccdtemp[0]
        else:
            raw[0] = hdus[camera][:, :]
        try:
            GFA.setraw(raw, name=camera)
        except ValueError as e:
            logging.error(e)
            return None
        GFA.data -= GFA.get_dark_current(ccdtemp, exptime)
        if camera.startswith('GUIDE'):
            GFA.get_psfs()
            stamps = GFA.psfs
            result = GFA.psf_stack
        else:
            GFA.get_donuts()
            stamps = GFA.donuts[0] + GFA.donuts[1]
            result = GFA.donut_stack
        if framepath is not None:
            label = '{0} {1} {2:.1f}s {3:.1f}C'.format(night, expid, exptime, ccdtemp)
            plot_data(GFA.data[0], GFA.ivar[0], downsampling=2, label=label, stamps=stamps, colorhist=True)
            plt.savefig(framepath / '{0}_{1:08d}.{2}'.format(camera, meta['EXPID'], img_format), quality=80)
            plt.clf()
        return result


def process(inpath, args, pool, pool_timeout=5):
    """Process a single GFA exposure.
    """
    if not inpath.exists():
        logging.error('Non-existant path: {0}'.format(inpath))
        return
    # Is this a guiding exposure?
    guiding = inpath.name.startswith('guide')
    # Lookup the NIGHT, EXPID, EXPTIME from the primary header.
    hdr = fitsio.read_header(str(inpath), ext='GUIDER' if guiding else 'GFA')
    for k in 'NIGHT', 'EXPID', 'EXPTIME':
        if k not in hdr:
            logging.info('Skipping exposure with missing {0}: {1}'.format(k, inpath))
            return
    night = str(hdr['NIGHT'])
    expid = '{0:08d}'.format(hdr['EXPID'])
    if hdr['EXPTIME'] == 0:
        logging.info('Skipping zero EXPTIME: {0}/{1}'.format(night, expid))
        return
    # Prepare the output path.
    outpath = args.outpath / night / expid
    outpath.mkdir(parents=True, exist_ok=True)
    # Are there already existing outputs?
    fitspath = outpath / 'stack_{0}.fits'.format(expid)
    if fitspath.exists() and not args.overwrite:
        logging.info('Will not overwrite outputs in {0}'.format(outpath))
        return
    # Process each camera in the input.
    logging.info('Processing {0}'.format(inpath))
    results = {}
    framepath = outpath if args.save_frames else None
    for camera in GFA.gfa_names:
        if guiding and camera.startswith('FOCUS'):
            # Guiding exposures do not record FOCUS data.
            continue
        # Fetch this camera's CCD temperatures and exposure times.
        if guiding:
            info = fitsio.read(str(inpath), ext=camera + 'T', columns=('EXPTIME', 'GCCDTEMP'))
        else:
            info = fitsio.read_header(str(inpath), ext=camera)
        exptime = info['EXPTIME']
        ccdtemp = info['GCCDTEMP']
        if pool is None:
            result = process_one(inpath, night, expid, guiding, camera, exptime, ccdtemp, framepath)
            if result is None:
                logging.error('Error processing HDU {0}'.format(camera))
            else:
                results[camera] = result
        else:
            results[camera] = pool.apply_async(
                process_one, (inpath, night, expid, guiding, camera, exptime, ccdtemp, framepath))
    if pool:
        # Collect the pooled results.
        for camera in results:
            try:
                result = results[camera].get(timeout=pool_timeout)
                if result is None:
                    logging.error('Error pool processing HDU {0}'.format(camera))
                else:
                    results[camera] = result
            except TimeoutError:
                logging.error('Timeout waiting for {0} pool result'.format(camera))
                del results[camera]
    # Save the output FITS file.
    with fitsio.FITS(str(fitspath), 'rw', clobber=True) as hdus:
        meta = {k: hdr.get(k) for k in (
            'NIGHT', 'EXPID', 'MJD-OBS', 'EXPTIME', 'HEXPOS', 'TRUSTEMP',
            'ADC1PHI', 'ADC2PHI', 'MOUNTHA', 'MOUNTAZ', 'MOUNTEL', 'MOUNTDEC')}
        hdus.write(np.zeros(1), header=meta)
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
    figpath = outpath / 'gfadiq_{0}.png'.format(expid)
    plt.savefig(figpath)
    plt.close(fig)
    logging.info('Wrote {0}'.format(figpath))


def get_gfa_exposures(inpath, checkpath, night, expstart=None, expstop=None):
    """Return a list of existing paths to completed GFA exposures for night.
    """
    paths = []
    # Has a directory for this night been created yet?
    checknight = checkpath / str(night)
    innight = inpath / str(night)
    if not checknight.exists() or not innight.exists():
        return paths
    # Loop over all exposure paths under this night.
    for exppath in checknight.glob('????????'):
        expid = str(exppath)[-8:]
        expnum = int(expid)
        if expstart is not None and expnum < expstart:
            continue
        if expstop is not None and expnum >= expstop:
            continue
        for pattern in 'gfa-{0}.fits.fz', 'guide-{0}.fits.fz':
            path = innight / expid / pattern.format(expid)
            if path.exists():
                paths.append(path)
    return set(paths)


def gfadiq():
    parser = argparse.ArgumentParser(
        description='Summarize delivered image quality for a GFA exposure.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--night', type=int, metavar='YYYYMMDD',
        help='Night of exposure to process in the format YYYYMMDD')
    parser.add_argument('--expid', type=str, metavar='N',
        help='Exposure(s) to process specified as N or N1-N2')
    parser.add_argument('--batch', action='store_true',
        help='Process all existing exposures on night')
    parser.add_argument('--watch', action='store_true',
        help='Wait for and process new exposures on night')
    parser.add_argument('--watch-interval', type=float, metavar='T', default=2,
        help='Interval in seconds to check for new exposures with --watch')
    parser.add_argument('--save-frames', action='store_true',
        help='Save images of each GFA frame')
    parser.add_argument('--overwrite', action='store_true',
        help='Overwrite existing outputs')
    parser.add_argument('--inpath', type=str, metavar='PATH',
        help='Path where raw data is organized under YYYYMMDD directories')
    parser.add_argument('--outpath', type=str, metavar='PATH',
        help='Path where outputs willl be organized under YYYYMMDD directories')
    parser.add_argument('--checkpath', type=str, metavar='PATH',
        help='Optional path where links are created to indicate a complete exposure')
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

    # Determine which directory to check for completed exposures.
    if args.checkpath is None:
        if host is 'DOS':
            args.checkpath = '/data/dts/exposures/raw/'
        else:
            args.checkpath = args.inpath
    if args.checkpath != args.inpath:
        if not args.checkpath.exists():
            print('Non-existant check path: {0}'.format(args.checkpath))
            sys.exit(-2)
        logging.info('Check path is {0}'.format(args.inpath))

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
        # Process an argument of the form N or N1-N2.
        limits = [int(expid) for expid in args.expid.split('-')]
        if len(limits) == 1:
            start, stop = limits[0], limits[0] + 1
        elif len(limits) == 2:
            start, stop = limits[0], limits[1] + 1
        else:
            print('Invalid --expid (should be N or N1-N2): "{0}"'.format(args.expid))
            sys.exit(-1)
        exposures = get_gfa_exposures(args.inpath, args.checkpath, args.night, start, stop)
        for path in exposures:
            process(path, args, pool)
        return

    if args.batch or args.watch:
        # Find the existing exposures on this night.
        existing = get_gfa_exposures(args.inpath, args.checkpath, args.night)
        if args.batch:
            for path in existing:
                process(path, args, pool)
        if args.watch:
            logging.info('Watching for new exposures...hit ^C to exit')
            try:
                while True:
                    time.sleep(args.watch_interval)
                    newexp = get_gfa_exposures(args.inpath, args.checkpath, args.night) - existing
                    for path in newexp:
                        process(path, args, pool)
                    existing |= newexp
            except KeyboardInterrupt:
                logging.info('Bye.')
                pass
