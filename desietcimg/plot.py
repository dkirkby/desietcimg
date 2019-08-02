"""Plotting utilities. Import requires matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches


def draw_ellipse_cov(ax, center, cov, nsigmas=2, **ellipseopts):
    U, s, _ = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * nsigmas * np.sqrt(s.T)
    kwargs = dict(color='w', ls='-', lw=2, fill=False)
    kwargs.update(ellipseopts)
    ellipse = matplotlib.patches.Ellipse(center, width, height, angle, **kwargs)
    ax.add_artist(ellipse)


def plot_image(D, W, cov=None, center=[0, 0], ax=None):
    data = D.copy()
    data[W == 0] = np.nan
    ax = ax or plt.gca()
    I = ax.imshow(data, interpolation='none', origin='lower')
    #plt.colorbar(I, ax=ax)
    if cov is not None:
        ny, nx = data.shape
        center = np.asarray(center) + 0.5 * (np.array(data.shape) - 1)
        draw_ellipse_cov(ax, center, cov)
    ax.axis('off')


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
