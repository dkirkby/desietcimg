"""Perform calibration analysis.
"""
import argparse
import sys
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fitsio

from desietcimg.calib import *
from desietcimg.util import *
from desietcimg.plot import *
from desietcimg.db import *


def getpath(expid, night, name='', tile=None, check_exists=True, root=Path('/project/projectdirs/desi/spectro/data/')):
    tag = f'{expid:08d}'
    path = root / str(night) / tag / name.format(tag=tag, tile=tile)
    if check_exists and not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def swap_cameras(hdus):
    """CIW,CIE and CIC,CIN are swapped before 02-Apr-2019.    
    This is a wrapper that undoes these swaps when accessing a camera HDU by name.
    """
    old_map = hdus.hdu_map
    new_map = dict(old_map)
    for old, new in ('ciw', 'cie'), ('cie', 'ciw'), ('cic', 'cin'), ('cin', 'cic'):
        if old in old_map:
            new_map[new] = old_map[old]
            if new not in old_map:
                del new_map[old]
    hdus.hdu_map = new_map
    return hdus


def openCI(expid, night, verbose=True):
    """
    """
    hdus = fitsio.FITS(getpath(expid, night, 'ci-{tag}.fits.fz'))
    # Check for an IMAGECAM header.
    hdr = hdus[1].read_header()
    if 'IMAGECAM' not in hdr:
        hdus.close()
        raise RuntimeError(f'Missing IMAGECAM in {path}.')
    # Check that each listed camera has an HDU present.
    missing = []
    expected = hdr['IMAGECAM'].split(',')
    for camera in expected:
        if camera not in hdus:
            missing.append(camera)
    if missing:
        hdus.close()
        raise RuntimeError(f'Missing HDU for {",".join(missing)} in {path}')
    # Fix CIW,CIE and CIC,CIN swaps before 02-Apr-2019.
    cutoff = 20190402
    if night < cutoff:
        if verbose:
            print(f'Swapping CIW,CIE and CIC,CIN for night {night} < {cutoff}.')
        hdus = swap_cameras(hdus)
    return hdus, hdr


def CIfiles(exposure_table, verbose=False):
    """Iterate over CI FITS files using a table containing `night` and `id` columns.
    """
    for name in 'id', 'night':
        if name not in exposure_table.columns:
            raise ValueError(f'Table has no "{name}" column.')
    for _, row in exposure_table.iterrows():
        night, expid = row['night'], row['id']
        if night is None or np.isnan(night) or (night < 20190317 or night > 20190701):
            print(f'Invalid night={night}.')
            continue
        # Pandas upcasts int column to float if it contains any invalid values.
        expid = int(round(expid))
        night = int(round(night))
        try:
            hdus, hdr = openCI(expid, night, verbose=verbose)
        except RuntimeError as e:
            print(e)
            continue
        # Check that header has consistent NIGHT and EXPID.
        if hdr['NIGHT'] != night:
            print(f'FITS header ({hdr["NIGHT"]} and db ({night}) have different NIGHT.')
            continue
        if hdr['EXPID'] != expid:
            print(f'FITS header ({hdr["EXPID"]} and db ({expid}) have different EXPID.')
            continue
        try:
            yield hdus, hdr, row
        finally:
            hdus.close()


def etccalib():
    parser = argparse.ArgumentParser(
        description='Analyze calibration data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--ci-night', type=int, default=0, metavar='YYYYMMDD',
        help='night of CI run to analyze in the format YYYYMMDD')
    parser.add_argument('--stxl-path', type=str, default='', metavar='PATH',
        help='path of STXL calibration files to analyze')
    parser.add_argument('--zero-min', type=int, default=None, metavar='N',
        help='only use zero exposures with sequence number >= N')
    parser.add_argument('--zero-max', type=int, default=None, metavar='N',
        help='only use zero exposures with sequence number <= N')
    parser.add_argument('--dark-min', type=int, default=None, metavar='N',
        help='only use dark exposures with sequence number >= N')
    parser.add_argument('--dark-max', type=int, default=None, metavar='N',
        help='only use dark exposures with sequence number <= N')
    parser.add_argument('--flat-min', type=int, default=None, metavar='N',
        help='only use flat exposures with sequence number >= N')
    parser.add_argument('--flat-max', type=int, default=None, metavar='N',
        help='only use flat exposures with sequence number <= N')
    parser.add_argument('--outpath', type=str, default='.',
        help='path where output files are saved')
    parser.add_argument('--binning', type=int, choices=(1, 2, 3), default=1, metavar='B',
        help='binning of STXL exposures')
    parser.add_argument('--db', type=str, default='db.yaml',
        help='yaml file of database connection parameters')
    args = parser.parse_args()

    outpath = Path(args.outpath)
    if args.outpath.endswith('.fits'):
        if args.ci_night > 0:
            print('Cannot specify filename in --outpath with --ci-night')
            sys.exit(-1)
        outpath = outpath.parent
    if not outpath.is_dir():
        print('Non-existant output path: {0}.'.format(outpath))
        sys.exit(-1)

    if args.stxl_path:
        inpath = Path(args.stxl_path)
        if not inpath.exists():
            raise FileNotFoundError(str(inpath))
        zero_paths = find_files(inpath / 'zero_{N}.fits', args.zero_min, args.zero_max)
        nzero = len(zero_paths)
        dark_paths = find_files(inpath / 'dark_{N}.fits', args.dark_min, args.dark_max)
        ndark = len(dark_paths)
        flat_paths = find_files(inpath / 'flat_{N}.fits', args.flat_min, args.flat_max)
        nflat = len(flat_paths)
        if args.verbose:
            print('Found {0} zero, {1} dark, {2} flat exposures from {3}'
                  .format(nzero, ndark, nflat, args.stxl_path))
        CA = CalibrationAnalysis('STXL', 2047 // args.binning, 3072 // args.binning)
        if nzero > 0:
            if args.verbose:
                print('Loading zero frames...')
            raw, meta = load_raw(zero_paths, 'EXPTIME', 'SET-TEMP', verbose=args.verbose)
            CA.process_zeros(raw, refine='auto', verbose=args.verbose)
        if nflat > 0:
            if args.verbose:
                print('Loading flat frames...')
            raw, meta = load_raw(flat_paths, 'SET-TEMP', verbose=args.verbose)
            downsampling = {1: 64, 2:32, 3:16}[args.binning]
            CA.process_flats(raw, downsampling=downsampling, verbose=args.verbose)
        if ndark > 0:
            if args.verbose:
                print('Loading dark frames...')
            raw, meta = load_raw(dark_paths, 'EXPTIME', 'SET-TEMP', verbose=args.verbose)
            CA.process_darks(raw, meta['SET-TEMP'], meta['EXPTIME'], verbose=args.verbose)
        # Save the results.
        if not args.outpath.endswith('.fits'):
            args.outpath = str(outpath / 'stxl-calib.fits')
        CA.save(args.outpath)

    if args.ci_night > 0:
        # Initialize the online database.
        db = DB(args.db)
        # Find all zero and dark exposures on this night.
        zero_exps = db.select(
            'exposure.exposure', 'id,night,exptime', limit=1000,
            where="sequence='CI' and flavor='zero' and night={0}".format(args.ci_night))
        nzero = len(zero_exps)
        dark_exps = db.select(
            'exposure.exposure', 'id,night,exptime', limit=1000,
            where="sequence='CI' and flavor='dark' and night={0}".format(args.ci_night))
        ndark = len(dark_exps)
        if args.verbose:
            print('Found {0} zero and {1} dark exposures from {2}'
                  .format(nzero, ndark, args.ci_night))
        # Process each camera separately. This is somewhat slower but uses 1/5 the memory.
        cameras = 'CIN', 'CIE', 'CIS', 'CIW', 'CIC'
        for camera in cameras:
            CA = CalibrationAnalysis(camera, 2048, 3072)
            if args.verbose:
                print('Loading {0} zero frames...'.format(camera))
            raw = np.empty((nzero,) + CA.shape, np.uint16)
            for k, (hdus, hdr, row) in enumerate(CIfiles(zero_exps)):
                raw[k] = hdus[camera].read()
            CA.process_zeros(raw, refine=False, verbose=args.verbose)
            if args.verbose:
                print('Loading {0} dark frames...'.format(camera))
            raw = np.empty((ndark,) + CA.shape, np.uint16)
            for k, (hdus, hdr, row) in enumerate(CIfiles(dark_exps)):
                raw[k] = hdus[camera].read()
            CA.process_darks(raw, verbose=args.verbose)
            CA.save(str(outpath / 'ci-calib-{0}-{1}.fits'.format(args.ci_night, camera)))
