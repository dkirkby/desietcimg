"""Simulate sky camera data and analysis.
"""
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from desietcimg.sky import *
from desietcimg.calib import *
from desietcimg.plot import *
from desietcimg.util import *


def simulate():
    parser = argparse.ArgumentParser(
        description='Simulate sky camera data and analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('-n', '--nstudy', type=int, default=1, metavar='N',
        help='number of exposures to simulate')
    parser.add_argument('--exptime', type=float, default=60,
        help='exposure time in seconds to simulate')
    parser.add_argument('--rate1', type=float, default=100,
        help='signal rate in elec/sec for the first fiber')
    parser.add_argument('--rate2', type=float, default=10.,
        help='signal rate in elec/sec for the last fiber')
    parser.add_argument('--logsteps', action='store_true',
        help='use log steps from rate1 to rate2 (otherwise linear)')
    parser.add_argument('--fibers', type=str, default='fibers.json',
        help='name of the JSON file with fiber locations to use')
    parser.add_argument('--calib', type=str,
        help='name of calibration analysis file to use')
    parser.add_argument('--bgraw', type=str, default=None,
        help='filename pattern for raw background images to use')
    parser.add_argument('--pixbias', action='store_true',
        help='simulate pixel variations in bias')
    parser.add_argument('--pixdark', action='store_true',
        help='simulate pixel variations in dark current')
    parser.add_argument('--pixmask', action='store_true',
        help='simulate pixel defects')
    parser.add_argument('--saveplot', type=str, default='simskycam.png',
        help='filename where plot of first simulated exposure is saved')
    parser.add_argument('--badplot', type=str, default=None,
        help='prefix for saving plots of bad fits')
    parser.add_argument('--outname', type=str, default='simskycam.csv',
        help='name of CSV file where results are saved when nstudy > 1')
    parser.add_argument('--seed', type=int, default=123,
        help='random seed to use')
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    # Use a separate state for the background so --bgraw  does not change the signal.
    bgrng = np.random.RandomState(args.seed)

    # Load the calibration analysis.
    CA = CalibrationAnalysis.load(args.calib)

    # Initialize an analyzer.
    SCA = SkyCameraAnalysis(CA)
    SCA.load_fiber_locations(args.fibers)

    if args.bgraw is not None:
        bgfiles = find_files(args.bgraw)
        if args.verbose:
            print('Found {0} files matching {1}.'.format(len(bgfiles), args.bgraw))
        # Load raw background images into memory. No need to load more images
        # than exposures we will simulate.
        bgraw, meta = load_raw(bgfiles[:args.nstudy], 'EXPTIME', 'SET-TEMP', verbose=args.verbose)
        nbgraw = len(bgraw)
        if meta['EXPTIME'] != args.exptime:
            print('WARNING: raw darks do not use requested EXPTIME.')
    else:
        nbgraw = 0

    if args.nstudy > 1:
        # Initialize the output file.
        labels = [label for label in SCA.fibers]
        simout = open(args.outname, 'w')
        print(*labels, sep=',', file=simout)

    # Calculate the true mean signal rates (elec/sec) in each fiber.
    true_means = init_signals(SCA.fibers, args.rate1, args.rate2, args.logsteps)

    # Loop over simulated exposures.
    for i in range(args.nstudy):

        if args.verbose and i % 100 == 0:
            print('Simulating {0} / {1}'.format(i + 1, args.nstudy))

        if nbgraw > 0:
            bg = bgraw[i % nbgraw]
        else:
            # Generate a random background image.
            bg = CA.simulate(exptime=args.exptime, pixbias=args.pixbias,
                             pixdark=args.pixdark, pixmask=args.pixmask, rng=bgrng)

        # Add random signals for each fiber.
        data, true_detected = add_fiber_signals(
            bg, true_means, SCA, args.exptime, CA.flatinvgain, rng=rng)

        # Perform the measurement.
        measured = SCA.get_fiber_fluxes(data, args.exptime)

        if i == 0 and args.verbose:
            for label in SCA.fibers:
                (xfit, yfit, bgmean, fiber_flux, snr, stamp, ivar, model, raw) = measured[label]
                print('{0}: mean={1:.1f} det={2:.1f} fit={3:.1f} SNR={4:.1f}'.format(
                    label, true_means[label], true_detected[label], fiber_flux, snr))

        if i == 0 and args.saveplot:
            A = plot_sky_camera(SCA, what='stamp')
            plt.savefig(args.saveplot)
            plt.close('all')

        if args.nstudy > 1:
            truth = [true_detected[label] for label in labels]
            # Save the estimated fiber fluxes.
            values = [measured[label][3] for label in labels]
            print(*values, sep=',', file=simout)
            if args.badplot is not None:
                # Save an image if any measurements are off by more than a factor of 2.
                ratio = np.array(values) / truth
                if np.any((ratio > 2) | (ratio < 0.5)):
                    name = '{0}_{1}.png'.format(args.badplot, i)
                    print('Saving bad fit to {0}'.format(name))
                    A = plot_sky_camera(SCA, what='stamp')
                    plt.savefig(name)
                    plt.close('all')

    if args.nstudy > 1:
        simout.close()
