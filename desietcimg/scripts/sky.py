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


def simulate():
    parser = argparse.ArgumentParser(
        description='Simulate sky camera data and analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--exptime', type=float, default=60,
        help='exposure time in seconds to simulate')
    parser.add_argument('--max-rate', type=float, default=150,
        help='max signal rate in elec/sec for the brightest fiber')
    parser.add_argument('--attenuation', type=float, default=0.95,
        help='signal attenuation between fibers')
    parser.add_argument('--fibers', type=str, default='fibers.json',
        help='name of the JSON file with fiber locations to use')
    parser.add_argument('--calib', type=str,
        help='name of calibration analysis file to use')
    parser.add_argument('--saveplot', type=str, default='simskycam.png',
        help='filename where plot is saved')
    parser.add_argument('--seed', type=int, default=123,
        help='random seed to use')
    args = parser.parse_args()

    # Load the calibration analysis.
    CA = CalibrationAnalysis.load(args.calib)

    # Initialize an analyzer.
    ny, nx = CA.shape
    if nx == 3072:
        binning = 1
    elif nx == 1536:
        binning = 2
    elif nx == 1024:
        binning = 3
    else:
        raise RuntimeError('Unable to infer binning from nx={0}.'.format(nx))
    SCA = desietcimg.sky.SkyCameraAnalysis(nx * binning, ny * binning, binning, CA.flatinvgain)
    SCA.load_fiber_locations(args.fibers)

    # Calculate the true mean signals (ADU) in each fiber.
    true_means = init_signals(SCA.fibers, args.max_rate, args.attenuation)

    # Calculate the background mean and RMS in ADU.
    dark = CA.dark_current * args.exptime / CA.flatinvgain
    bgmean = CA.avgbias + dark
    bgrms = np.sqrt(CA.rdnoise ** 2 + CA.flatinvgain * CA.dark_current)

    # Generate a random background frame.
    rng = np.random.RandomState(args.seed)
    ny, nx = SCA.ny // SCA.binning, SCA.nx // SCA.binning
    bg = rng.normal(loc=bgmean, scale=bgrms, size=(ny, nx))

    # Add random signals for each fiber.
    data, true_detected = add_fiber_signals(
        bg, true_means, SCA, args.exptime, CA.flatinvgain, rng=rng)

    # Perform the measurement.
    measured = SCA.get_fiber_fluxes(data, args.exptime)

    if args.verbose:
        for label in SCA.fibers:
            (xfit, yfit, bgmean, fiber_flux, snr, stamp) = measured[label]
            print('{0}: mean={1:.1f} det={2:.1f} fit={3:.1f} SNR={4:.1f}'.format(
                label, true_means[label], true_detected[label], fiber_flux, snr))

    if args.saveplot:
        A = plot_sky_camera(SCA)
        plt.savefig(args.saveplot)
