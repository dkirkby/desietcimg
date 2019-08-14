"""Plotting utilities. Import requires matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.cm


def draw_ellipse(ax, x0, y0, s, g1, g2, nsigmas=1, **ellipseopts):
    g = np.sqrt(g1 ** 2 + g2 ** 2)
    if g > 1:
        raise ValueError('g1 ** 2 + g2 ** 2 > 1')
    center = np.array([x0, y0])
    angle = np.rad2deg(0.5 * np.arctan2(g2, g1))
    ratio = np.sqrt((1 + g) / (1 - g))
    width = 2 * s * ratio * nsigmas
    height  = 2 * s / ratio * nsigmas
    kwargs = dict(color='r', ls='-', lw=2, alpha=0.7, fill=False)
    kwargs.update(ellipseopts)
    ellipse = matplotlib.patches.Ellipse(center, width, height, angle, **kwargs)
    ax.add_artist(ellipse)


def plot_image(D, W=None, ax=None, cmap='viridis', masked_color='chocolate'):
    if W is not None and np.any(W == 0):
        D = D.copy()
        D[W == 0] = np.nan
    ax = ax or plt.gca()
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad(color=masked_color)
    h, w = D.shape
    I = ax.imshow(D, interpolation='none', origin='lower', cmap=cmap,
                  extent=[-0.5 * w, 0.5 * w, -0.5 * h, 0.5 * h])
    ax.axis('off')
    return ax


class Axes(object):
    
    def __init__(self, n, size=4, pad=0.02):
        rows = int(np.floor(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        assert rows * cols >= n
        width = cols * size + (cols - 1) * pad
        height = rows * size + (rows - 1) * pad
        self.fig, axes = plt.subplots(rows, cols, figsize=(width, height))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=pad, hspace=pad)
        self.axes = axes.flatten()
        self.n = n
        for ax in self.axes:
            ax.axis('off')


def plot_sky_camera(SCA, size=4, pad=0.02, labels=True, params=True):
    if SCA.fibers is None or SCA.results is None:
        raise RuntimeError('No results available to plot.')
    nfibers = len(SCA.fibers)
    A = Axes(nfibers, size, pad)
    results = iter(SCA.results.items())
    fibers = iter(SCA.fibers.values())
    # Use the same colorscale for all stamps.
    allpixels = np.concatenate([result[-1] for result in SCA.results.values()], axis=1).flatten()
    vmin, vmax = np.percentile(allpixels, (1, 99))
    for k in range(nfibers):
        ax = A.axes[k]
        ix, iy = next(fibers)
        label, (xfit, yfit, bgmean, fiber_flux, snr, stamp) = next(results)
        ax.imshow(stamp, interpolation='none', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        cx = (xfit - ix) / SCA.binning + SCA.rsize
        cy = (yfit - iy) / SCA.binning + SCA.rsize
        cr = 0.5 * SCA.fiberdiam / SCA.binning
        circle = matplotlib.patches.Circle(
            [cx, cy], cr, lw=2, ec='r', fc='none', alpha=0.5)
        dxy = SCA.dxy + SCA.rsize
        xgrid, ygrid = np.meshgrid(dxy, dxy)
        ax.plot(xgrid[0], ygrid[0], 'r.', ms=1)
        ax.plot(xgrid[-1], ygrid[-1], 'r.', ms=1)
        ax.plot(xgrid[:, 0], ygrid[:, 0], 'r.', ms=1)
        ax.plot(xgrid[:, -1], ygrid[:, -1], 'r.', ms=1)
        ax.plot(cx, cy, 'r+')
        ax.add_artist(circle)
        kwargs = dict(verticalalignment='center', horizontalalignment='center',
                      transform=ax.transAxes, color='w', fontweight='bold')
        if labels:
            ax.text(0.5, 0.95, label, fontsize=16, **kwargs)
        if params:
            params = '{0:.1f} ADU SNR {1:.1f}'.format(fiber_flux, snr)
            ax.text(0.5, 0.05, params, fontsize=14, **kwargs)
    return A


def plot_guide_results(GCR, size=4, pad=0.02, ellipses=True, params=True):
    if GCR.stamps is None or GCR.results is None:
        raise RuntimeError('No results available to plot.')
    nstamps = GCR.meta['NSRC']
    rsize = GCR.meta['SSIZE'] // 2
    A = Axes(nstamps, size, pad)
    for k in range(nstamps):
        ax = A.axes[k]
        plot_image(*GCR.stamps[k], ax=ax)
        kwargs = dict(verticalalignment='center', horizontalalignment='center',
                      transform=ax.transAxes, fontweight='bold')
        result, y_slice, x_slice = GCR.results[k]
        ix, iy = x_slice.start + rsize, y_slice.start + rsize
        label = 'x={0:04d} y={1:04d}'.format(ix, iy)
        ax.text(0.5, 0.05, label, fontsize=16, color='w', **kwargs)
        if result['success']:
            color = 'w' if result['psf'] else 'r'
            if ellipses:
                draw_ellipse(ax, result['x0'], result['y0'],
                             result['s'], result['g1'], result['g2'], ec=color)
            if params:
                g = np.sqrt(result['g1'] ** 2 + result['g2'] ** 2)
                label = f'$\\nu$ {result["snr"]:.1f} s {result["s"]:.1f} g {g:.2f}'
                ax.text(0.5, 0.95, label, fontsize=18, color=color, **kwargs)
    return A


def plot_psf_profile(GCR, fiberdiam_um=107, pixel_size_um=9., inset_size=35):
    assert inset_size % 2 == 1
    P, W = GCR.profile
    h1 = len(P) // 2
    h2 = inset_size // 2
    inset = slice(h1 - h2, h1 + h2 + 1)
    ax = plot_image(P[inset, inset], W[inset, inset])
    rfiber_pix = 0.5 * fiberdiam_um / pixel_size_um
    ax.add_artist(plt.Circle((0, 0), rfiber_pix, fc='none', ec='r', lw=2, alpha=0.5))
