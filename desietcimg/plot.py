"""Plotting utilities. Import requires matplotlib.
"""
import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.cm

import desietcimg.util


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


def plot_image(D, W=None, ax=None, cmap='viridis', masked_color='chocolate', threshold=0.01):
    if W is not None and np.any(W == 0):
        D = D.copy()
        D[W == 0] = np.nan
    if W is not None:
        # Ignore pixels with low ivar for setting color scale limits.
        informative = W > threshold * np.median(W)
    else:
        informative = np.ones_like(D, bool)
    vmin, vmax = np.percentile(D[informative], (0, 100))
    ax = ax or plt.gca()
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad(color=masked_color)
    h, w = D.shape
    I = ax.imshow(D, interpolation='none', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
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


def plot_psf_profile(GCR, size=4, pad=0.5, inset_size=35, max_ang=2.0, label=None):
    """
    """
    assert inset_size % 2 == 1
    P, W = GCR.profile
    h1 = len(P) // 2
    h2 = inset_size // 2
    inset = slice(h1 - h2, h1 + h2 + 1)

    width = 2.5 * size + pad
    height = size
    fig = plt.figure(figsize=(width, height))
    lhs = plt.axes((0., 0., size / width, 1.))
    rhs = plt.axes(((size + pad) / width, pad / height,
                    (width - size - pad) / width - 0.02, (height - pad) / height - 0.02))
    plot_image(P[inset, inset], W[inset, inset], ax=lhs)
    kwargs = dict(fontsize=16, color='w', verticalalignment='center', horizontalalignment='center',
                  transform=lhs.transAxes, fontweight='bold')
    fwhm = GCR.meta['FWHM']
    ffrac = GCR.meta['FFRAC']
    xc = GCR.meta['XC']
    yc = GCR.meta['YC']
    lhs.text(0.5, 0.95, 'FWHM={0:.2f}"  FFRAC={1:.3f}'.format(fwhm, ffrac), **kwargs)
    if label is not None:
        lhs.text(0.5, 0.05, label, **kwargs)
    rfiber_pix = 0.5 * GCR.meta['FIBSIZ'] / GCR.meta['PIXSIZ']
    lhs.add_artist(plt.Circle((xc, yc), rfiber_pix, fc='none', ec='r', lw=2, alpha=0.5))
    lhs.plot(xc, yc, 'r+', ms=25)
    lhs.plot([xc, h2], [yc, yc], 'r--')
    lhs.plot([xc, xc], [yc, h2], 'r:')
    # Plot the circularized radial profile.
    rhs.plot(GCR.profile_tab['rang'], GCR.profile_tab['prof'], 'k-', label='Circ. Profile')
    # Plot the fiber acceptance fraction for centroid offsets along +x and +y.
    noffset = len(GCR.fiberfrac)
    noffset_per_pix = GCR.meta.get('NOFFPX', 2)
    dxy_pix = (np.arange(noffset) - 0.5 * (noffset - 1)) / noffset_per_pix
    pixel_size_um = GCR.meta['PIXSIZ']
    plate_scales = (GCR.meta['XSCALE'], GCR.meta['YSCALE'])
    dx_ang = (dxy_pix - xc) * pixel_size_um / plate_scales[0]
    dy_ang = (dxy_pix - yc) * pixel_size_um / plate_scales[0]
    iy, ix = np.unravel_index(np.argmax(GCR.fiberfrac), GCR.fiberfrac.shape)
    rhs.plot(dx_ang[ix:], GCR.fiberfrac[iy, ix:], 'r--', label='Fiber Frac (x)')
    rhs.plot(dx_ang[iy:], GCR.fiberfrac[iy:, ix], 'r:', label='Fiber Frac (y)')
    rhs.set_ylim(-0.02, 1.02)
    rhs.set_xlim(0., max_ang)
    rhs.grid()
    rhs.legend(loc='upper right')
    rhs.set_xlabel('Offset from PSF center [arcsec]')


def plot_full_frame(D, W=None, downsampling=8, clip_pct=0.5, dpi=100, GCR=None,
                    label=None, cmap='plasma_r', fg_color='w'):
    # Convert to a float32 array.
    D, W = desietcimg.util.prepare(D, W)
    # Downsample.
    WD = desietcimg.util.downsample(D * W, downsampling=downsampling, summary=np.sum, allow_trim=True)
    W = desietcimg.util.downsample(W, downsampling=downsampling, summary=np.sum, allow_trim=True)
    D = np.divide(WD, W, out=np.zeros_like(WD), where=W > 0)
    # Select background pixels using sigma clipping.
    sel = W > 0 if W is not None else (slice(None), slice(None))
    clipped, _, _ = scipy.stats.sigmaclip(D[sel])
    # Subtract the clipped mean from the data.
    bgmean = np.mean(clipped)
    bgrms = np.std(clipped)
    Z = np.arcsinh((D - bgmean) / bgrms)
    vmin, vmax = np.percentile(Z.reshape(-1), (clip_pct, 100 - clip_pct))
    ny, nx = D.shape
    fig = plt.figure(figsize=(nx / dpi, ny / dpi), dpi=dpi, frameon=False)
    ax = plt.axes((0, 0, 1, 1))
    extent = [-0.5, nx * downsampling - 0.5, -0.5, ny * downsampling - 0.5]
    ax.imshow(Z, interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
              cmap=cmap, extent=extent)
    ax.axis('off')
    if GCR is not None:
        for fit, yslice, xslice in GCR.results:
            xlo, xhi = xslice.start, xslice.stop
            ylo, yhi = yslice.start, yslice.stop
            ax.add_artist(plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                        fc='none', ec=fg_color, alpha=0.75))
    if label is not None:
        ax.text(0.01, 0.01 * nx / ny, label, color=fg_color, fontsize=18, transform=ax.transAxes)
    return ax


def plot_calib_frame(CA, what='bias', downsampling=4, cmap='viridis', masked_color='chocolate', dpi=100):
    ny, nx = CA.shape
    fig = plt.figure(figsize=(nx / downsampling / dpi, ny / downsampling / dpi), dpi=dpi, frameon=False)
    ax = plt.axes((0, 0, 1, 1))
    ax.axis('off')
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad(color=masked_color)
    if what in ('bias', 'mu'):
        D = CA.pixbias.copy() if what == 'bias' else CA.pixmu.copy()
        W = np.array(CA.pixmask == 0).astype(np.float32)
        WDds = desietcimg.util.downsample(D * W, downsampling, np.sum, allow_trim=True)
        Wds = desietcimg.util.downsample(W, downsampling, np.sum, allow_trim=True)
        Dds = np.divide(WDds, Wds, out=np.zeros_like(WDds), where=Wds > 0)
        vmin, vmax = np.percentile(Dds[Wds > 0], (0.5, 99.5))
        Dds[Wds <= 0] = np.nan
        ax.imshow(Dds, interpolation='none', origin='lower', cmap=cmap)
    elif what == 'mask':
        ax.imshow(CA.pixmask, interpolation='none', origin='lower')


def plot_calib_data(CA, what='zero', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if what in ('zero_fit', 'dark_fit'):
        data = CA.zerodata if what == 'zero' else CA.darkdata
        axplot(data['xpix'], data['ydata'], 'k.', label='Data')
        ax.plot(data['xpix'], data['yfit'], 'b-', alpha=0.5, label='Fit')
        ax.set_yticks([])
        ax.set_xlabel('Pixel Value [ADU]')
        plt.legend()
    elif what in ('pixbias', 'pixmu'):
        data = CA.pixbias if what == 'pixbias' else CA.pixmu
        data = data[CA.pixmask == 0]
        bins = np.linspace(*np.percentile(data, (0.5, 99.5)), 51)
        ax.hist(data, bins)
        mean = np.mean(data)
        ax.axvline(mean, ls='--', c='k', label='Mean {0:.1f} ADU'.format(mean))
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel('Pixel Mean [ADU]')
        ax.set_yticks([])
        plt.legend()
