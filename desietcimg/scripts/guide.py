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


db = None
ExpInfo = None

def get_pm_path(expid, night=None, dbname='db.yaml', root=Path('/project/projectdirs/desi/spectro/data/')):
    global db, ExpInfo
    if night is None:
        if db is None:
            # Initialize the exposure db.
            db = DB(dbname)
            ExpInfo = Exposures(db)
        # Lookup the night for this sequence number.
        night = ExpInfo(expid, 'night')
    tag = '{0:08d}'.format(expid)
    path = root / str(night) / tag / 'guide-{0}.fits.fz'.format(tag)
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


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
    parser.add_argument('--saveimg', action='store_true',
        help='save a downsampled full frame image showing selected stamps')
    parser.add_argument('--downsampling', type=int, default=8,
        help='downsampling to use for full frame images')
    parser.add_argument('--outpath', type=str, default='.',
        help='path where output files are saved')
    parser.add_argument('--db', type=str, default='db.yaml',
        help='yaml file of database connection parameters')
    parser.add_argument('input', nargs='+', type=str,
        help='FITS file, expid, or .dat file containing a list of expids')
    args = parser.parse_args()

    GCA = GuideCameraAnalysis(stamp_size=args.stamp_size)

    warnings.simplefilter(action='ignore', category=FutureWarning)

    outpath = Path(args.outpath)

    # Build the list of files to process from the input arguments.
    files = []
    for input in args.input:
        try:
            expid = int(input)
            files.append(get_pm_path(expid, dbname=args.db))
            continue
        except ValueError:
            pass
        if not os.path.exists(input):
            print('No such file: {0}.'.format(input))
            continue
        if input.endswith('.dat'):
            # Read a list of sequence numbers to process.
            with open(input, 'r') as f:
                for line in f.readlines():
                    expid = int(line.strip())
                    files.append(get_pm_path(expid, dbname=args.db))
            continue
        elif input.endswith('.fits') or input.endswith('.fits.fz'):
            files.append(input)

    # Loop over files.
    for file in files:
        with fitsio.FITS(file, 'r') as hdus:
            meta = hdus[0].read_header()
            night = str(meta['NIGHT'])
            expid = '{0:08d}'.format(meta['EXPID'])
            print('Processing {0} from {1}...'.format(expid, night))
            for camera in 'CIN', 'CIE', 'CIS', 'CIW', 'CIC':
                if camera in hdus:
                    D = hdus[camera][0, :, :][0]
                    GCR = GCA.detect_sources(D, meta=meta, nsrc=args.nsrc)
                    if args.verbose:
                        print('== {0}:'.format(camera))
                        GCR.print()
                    path = outpath / night / expid
                    path.mkdir(parents=True, exist_ok=True)
                    output = path / 'GCR-{0}-{1}.fits'.format(expid, camera)
                    GCR.save(str(output))
                    if args.saveimg:
                        plot_full_frame(D, GCR=GCR, label='{0}-{1}'.format(expid, camera),
                                        downsampling=args.downsampling)
                        output = path / 'FULL-{0}-{1}.png'.format(expid, camera)
                        plt.savefig(str(output))
                        plt.close('all')
