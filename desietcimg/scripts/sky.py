"""Simulate sky camera data and analysis.
"""
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from desietcimg.sky import *
from desietcimg.plot import *


def simulate():
    parser = argparse.ArgumentParser(
        description='Simulate sky camera data and analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--max-signal', type=float, default=5000,
        help='mean signal in ADU for the brightest fiber')
    parser.add_argument('--attenuation', type=float, default=0.95,
        help='signal attenuation between fibers')
    parser.add_argument('--binning', type=int, choices=(1, 2, 3), default=1,
        help='camera readout binning to simulate')
    parser.add_argument('--fibers', type=str, default='fibers.json',
        help='name of the JSON file with fiber locations to use')
    parser.add_argument('--bias', type=float, default=1000,
        help='mean bias level in ADU')
    parser.add_argument('--bgrms', type=float, default=13,
        help='RMS background fluctuations in ADU')
    parser.add_argument('--saveplot', type=str, default='simskycam.png',
        help='filename where plot is saved')
    parser.add_argument('--seed', type=int, default=123,
        help='random seed to use')
    args = parser.parse_args()

    # Initialize an analyzer.
    SCA = SkyCameraAnalysis(binning=args.binning)
    SCA.load_fiber_locations(args.fibers)

    # Generate a random background frame.
    rng = np.random.RandomState(args.seed)
    ny, nx = SCA.ny // SCA.binning, SCA.nx // SCA.binning
    bg = rng.normal(loc=args.bias, scale=args.bgrms, size=(ny, nx))

    # Calculate the true mean signals (ADU) in each fiber.
    true_means = init_signals(SCA.fibers, args.max_signal, args.attenuation)

    # Add random signals for each fiber.
    data, true_detected = add_fiber_signals(bg, true_means, SCA, rng=rng)

    # Perform the measurement.
    measured = SCA.get_fiber_fluxes(data)

    if args.verbose:
        for label in SCA.fibers:
            (xfit, yfit, bgmean, fiber_flux, snr, stamp) = measured[label]
            print('{0}: mean={1:.1f} det={2:.1f} fit={3:.1f} SNR={4:.1f}'.format(
                label, true_means[label], true_detected[label], fiber_flux, snr))

    if args.saveplot:
        A = plot_sky_camera(SCA)
        plt.savefig(args.saveplot)
