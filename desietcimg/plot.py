"""Plotting utilities. Import requires matplotlib.
"""
import datetime
import copy # for shallow copies of matplotlib colormaps

import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.lines
import matplotlib.patheffects
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
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
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
        self.fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=pad, hspace=pad)
        self.axes = axes.flatten()
        self.n = n
        for ax in self.axes:
            ax.axis('off')


def plot_sky_camera(SCA, size=4, pad=0.02, what='stamp', labels=True, params=True, fiber=True):
    if SCA.fibers is None or SCA.results is None:
        raise RuntimeError('No results available to plot.')
    nfibers = len(SCA.fibers)
    A = Axes(nfibers, size, pad)
    # Extract the pixel values to plot.
    plotdata = []
    results = iter(SCA.results.items())
    for k in range(nfibers):
        label, (xfit, yfit, bgmean, fiber_flux, snr, stamp, ivar, model, raw) = next(results)
        plotdata.append({'stamp': stamp, 'ivar': ivar, 'model': model, 'raw': raw}[what])
    # Use the same colorscale for all stamps.
    allpixels = np.concatenate(plotdata, axis=1).flatten()
    vmin, vmax = np.percentile(allpixels[allpixels > 0], (1, 99))
    # Loop over fibers to plot.
    results = iter(SCA.results.items())
    fibers = iter(SCA.fibers.values())
    for k in range(nfibers):
        ax = A.axes[k]
        ix, iy = next(fibers)
        label, (xfit, yfit, bgmean, fiber_flux, snr, stamp, ivar, model, raw) = next(results)
        ax.imshow(plotdata[k], interpolation='none', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if fiber:
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
            params = '{0:.1f} e/s SNR {1:.1f}'.format(fiber_flux, snr)
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


def plot_colorhist(D, ax, imshow, mode='reverse', color='w', alpha=0.75):
    """Draw a hybrid colorbar and histogram.
    """
    ax.axis('off')
    # Extract parameters of the original imshow.
    cmap = imshow.get_cmap()
    vmin, vmax = imshow.get_clim()
    # Get the pixel dimension of the axis to fill.
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = int(round(bbox.width * fig.dpi)), int(round(bbox.height * fig.dpi))
    # Draw the colormap gradient.
    img = np.zeros((height, width, 3))
    xgrad = np.linspace(0, 1, width)
    img[:] = cmap(xgrad)[:, :-1]
    # Superimpose a histogram of pixel values.
    counts, _ = np.histogram(D.reshape(-1), bins=np.linspace(vmin, vmax, width + 1))
    hist_height = ((height - 1) * counts / counts[1:-1].max()).astype(int)
    mask = np.arange(height).reshape(-1, 1) < hist_height
    if mode == 'color':
        img[mask] = (1 - alpha) * img[mask] + alpha * np.asarray(matplotlib.colors.to_rgb(color))
    elif mode == 'reverse':
        cmap_r = cmap.reversed()
        for i, x in enumerate(xgrad):
            img[mask[:, i], i] = cmap_r(x)[:-1]
    elif mode == 'complement':
        # https://stackoverflow.com/questions/40233986/
        # python-is-there-a-function-or-formula-to-find-the-complementary-colour-of-a-rgb
        hilo = np.amin(img, axis=2, keepdims=True) + np.amax(img, axis=2, keepdims=True)
        img[mask] = hilo[mask] - img[mask]
    else:
        raise ValueError('Invalid mode "{0}".'.format(mode))
    ax.imshow(img, interpolation='none', origin='lower')


def plot_pixels(D, label=None, colorhist=False, zoom=1, masked_color='cyan',
                imshow_args={}, text_args={}, colorhist_args={}):
    """Plot pixel data at 1:1 scale with an optional label and colorhist.
    """
    dpi = 100 # value only affects metadata in an output file, not appearance on screen.
    ny, nx = D.shape
    width, height = zoom * nx, zoom * ny
    if colorhist:
        colorhist_height = 32
        height += colorhist_height
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = plt.axes((0, 0, 1, zoom * ny / height))
    args = dict(imshow_args)
    for name, default in dict(interpolation='none', origin='lower', cmap='plasma_r').items():
        if name not in args:
            args[name] = default
    # Set the masked color in the specified colormap.
    cmap = copy.copy(matplotlib.cm.get_cmap(args['cmap']))
    cmap.set_bad(color=masked_color)
    args['cmap'] = cmap
    # Draw the image.
    I = ax.imshow(D, **args)
    ax.axis('off')
    if label:
        args = dict(text_args)
        for name, default in dict(color='w', fontsize=18).items():
            if name not in args:
                args[name] = default
        outline = [
            matplotlib.patheffects.Stroke(linewidth=1, foreground='k'),
            matplotlib.patheffects.Normal()]
        text = ax.text(0.01, 0.01 * nx / ny, label, transform=ax.transAxes, **args)
        text.set_path_effects(outline)
    if colorhist:
        axcb = plt.axes((0, zoom * ny / height, 1, colorhist_height / height))
        plot_colorhist(D, axcb, I, **colorhist_args)
    return fig, ax


def plot_data(D, W, downsampling=4, zoom=1, label=None, colorhist=False, stamps=[],
              preprocess_args={}, imshow_args={}, text_args={}, colorhist_args={}):
    """Plot weighted image data using downsampling, optional preprocessing, and decorators.
    """
    # Downsample the input data.
    D, W = desietcimg.util.downsample_weighted(D, W, downsampling)
    # Preprocess the data for display.
    D = desietcimg.util.preprocess(D, W, **preprocess_args)
    ny, nx = D.shape
    # Display the image.
    args = dict(imshow_args)
    if 'extent' not in args:
        # Use the input pixel space for the extent, without downsampling.
        args['extent'] = [-0.5, nx * downsampling - 0.5, -0.5, ny * downsampling - 0.5]
    fig, ax = plot_pixels(D, zoom=zoom, label=label, colorhist=colorhist,
                          imshow_args=args, text_args=text_args, colorhist_args=colorhist_args)
    outline = [
        matplotlib.patheffects.Stroke(linewidth=1, foreground='k'),
        matplotlib.patheffects.Normal()]
    for k, stamp in enumerate(stamps):
        yslice, xslice = stamp[:2]
        xlo, xhi = xslice.start, xslice.stop
        ylo, yhi = yslice.start, yslice.stop
        rect = plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fc='none', ec='w', lw=1)
        ax.add_artist(rect)
        if xhi < nx // 2:
            xtext, halign = xhi, 'left'
        else:
            xtext, halign = xlo, 'right'
        text = ax.text(
            xtext, 0.5 * (ylo + yhi), str(k), fontsize=12, color='w', va='center', ha=halign)
        text.set_path_effects(outline)
    return fig, ax


def plot_full_frame(D, W=None, saturation=None, downsampling=8, clip_pct=0.5, dpi=100, GCR=None,
                    label=None, cmap='plasma_r', fg_color='w', compress=True, vmin=None, vmax=None):
    # Convert to a float32 array.
    D, W = desietcimg.util.prepare(D, W, saturation=saturation)
    # Downsample.
    WD = desietcimg.util.downsample(D * W, downsampling=downsampling, summary=np.sum, allow_trim=True)
    W = desietcimg.util.downsample(W, downsampling=downsampling, summary=np.sum, allow_trim=True)
    D = np.divide(WD, W, out=np.zeros_like(WD), where=W > 0)
    # Used clipped limits by default.
    clip_vmin, clip_vmax = np.percentile(D.reshape(-1), (clip_pct, 100 - clip_pct))
    # Any explicit limit overrides the clipped default.
    vmin = vmin or clip_vmin
    vmax = vmax or clip_vmax
    if vmin >= vmax:
        raise ValueError('Invalid limits: vmin >= vmax.')
    if compress:
        # Select background pixels using sigma clipping.
        sel = W > 0 if W is not None else (slice(None), slice(None))
        clipped, _, _ = scipy.stats.sigmaclip(D[sel])
        # Subtract the clipped mean from the data.
        bgmean = np.mean(clipped)
        bgrms = np.std(clipped)
        Z = np.arcsinh((D - bgmean) / bgrms)
        vmin = np.arcsinh((vmin - bgmean) / bgrms)
        vmax = np.arcsinh((vmax - bgmean) / bgrms)
    else:
        Z = D
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
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    cmap.set_bad(color=masked_color)
    if what in ('bias', 'mu'):
        D = CA.pixbias.copy() if what == 'bias' else CA.darkmu.copy()
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
        data = CA.pixbias if what == 'pixbias' else CA.darkmu
        data = data[CA.pixmask == 0]
        bins = np.linspace(*np.percentile(data, (0.5, 99.5)), 51)
        ax.hist(data, bins)
        mean = np.mean(data)
        ax.axvline(mean, ls='--', c='k', label='Mean {0:.1f} ADU'.format(mean))
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel('Pixel Mean [ADU]')
        ax.set_yticks([])
        plt.legend()


def plot_gain_fit(CA, ax=None):
    if not CA.have_flats:
        raise ValueError('Calibration does not include flats analysis.')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(CA.flatdata['mu'], CA.flatdata['var'], 'k+')
    xmax = 1.05 * CA.flatdata['mu'].max()
    g = CA.flatinvgain
    ax.plot([0, xmax], [0, xmax / g], 'r--', label='g = {0:.2f} e/ADU'.format(g))
    ax.legend()
    ax.set_xlabel('Bias subtracted signal [ADU]')
    ax.set_ylabel('Read-noise subtracted variance [ADU$^2$]')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)


def plot_dark_calib(CA, gain=1.5, peaklines=True, lo=None, hi=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Plot second x-axis in units of dark current.
    conv = gain / CA.dark_exptime
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlabel('Dark current at {0:.0f}C [e/sec/pix]'.format(CA.dark_temperature))

    xbin = CA.darkdata['xbin']
    if lo is None:
        lo = xbin[0]
    if hi is None:
        hi = xbin[-1]

    ax.plot(xbin, CA.darkdata['yexp'], 'k-', label='Single Exp.')
    ax.plot(xbin, CA.darkdata['yavg'], 'r-', label='Stack of {0}'.format(CA.dark_nexp))
    ax.plot(xbin, CA.darkdata['yfit'], 'r--', label='Model Fit')

    if peaklines:
        # Draw lines at the best-fit peak locations.
        x0, navg, spacing, c0, c1, c2 = CA.darktheta
        for n in range(5):
            x = x0 + spacing * n
            ax.axvline(x, ls=':', c='gray')

    ax.set_xlim(lo, hi)
    ax2.set_xlim(conv * lo, conv * hi)
    ax.set_ylim(0, None)
    ax.set_yticks([])
    ax.set_xlabel('Bias subtracted pixel value [ADU]')
    ax.legend(title='{0} @ {1:.0f}s'.format(CA.name, CA.dark_exptime))


def plot_stack(stack, cmap='plasma_r', masked_color='cyan', offset=0):
    nstack = len(stack)
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    cmap.set_bad(color=masked_color)
    A = desietcimg.plot.Axes(nstack, size=2, pad=0.01)
    outline = [
        matplotlib.patheffects.Stroke(linewidth=1, foreground='k'),
        matplotlib.patheffects.Normal()]
    for k, (ax, (_, _, D, W)) in enumerate(zip(A.axes[:nstack], stack)):
        mu = np.sum(D * W) / np.sum(W)
        ivar = np.median(W[W > 0])
        z = np.arcsinh((D - mu) * (0.02 * np.sqrt(ivar)))
        vmin, vmax = np.percentile(z[W > 0], (1, 99))
        z[W == 0] = np.nan
        ax.imshow(z, interpolation='none', origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        text = ax.text(0.5, 0, str(k + offset), transform=ax.transAxes, fontsize=14,
                       color='w', va='bottom', ha='center')
        text.set_path_effects(outline)


def plot_distance_matrix(stamps, cmap='magma', masked_color='cyan', dpi=100, maxsize=1024,
                         smoothing=1, maxdither=3, maxdist=3.):
    nstamps = len(stamps)
    # Extract and normalize stamps.
    stamps = [desietcimg.util.normalize_stamp(*S[2:4]) for S in stamps]
    # Calculate the median stamp.
    Dmed = np.median(np.stack([D for D, W in stamps]), axis=0)
    # Calculate the weighted average stamp.
    DWsum = np.sum(np.stack([D * W for D, W in stamps]), axis=0)
    Wavg = np.sum(np.stack([W for D, W in stamps]), axis=0)
    Davg = np.divide(DWsum, Wavg, out=np.zeros_like(DWsum), where=Wavg > 0)
    Davg, Wavg = desietcimg.util.normalize_stamp(Davg, Wavg)
    # Calculate plot limits from the median stamp.
    vmin, vmax = np.percentile(Dmed, (1, 99))
    # Initialize figure.
    ny, nx = (stamps[0][0]).shape
    assert ny == nx
    ndither = 2 * maxdither + 1
    zoom = max(1, maxsize // (nstamps * (ny - ndither)))
    dxy = (ny - ndither) * zoom
    size = nstamps * dxy
    inset = slice(maxdither, ny - maxdither), slice(maxdither, nx - maxdither)
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, frameon=False)
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    cmap.set_bad(color=masked_color)
    def create_axis(i, j, expand=1):
        ax = plt.axes((i * dxy / size, j * dxy / size, expand * dxy / size, expand * dxy / size))
        ax.axis('off')
        return ax
    def imshow(z, ax, label, label_color, **args):
        ax.imshow(z, interpolation='none', origin='lower', **args)
        ax.text(0.5, 0, label, transform=ax.transAxes, fontsize=10, color=label_color,
                verticalalignment='bottom', horizontalalignment='center')
    def plot_stamp(D, W, ax, label):
        d = D.copy()
        if W is not None:
            d[W == 0] = 0
        imshow(d, ax, label, 'w', vmin=vmin, vmax=vmax, cmap=cmap)
    def plot_pull(pull, ax, vlim=5):
        chisq = np.sum(pull ** 2) / pull.size
        imshow(pull, ax, f'{chisq:.2f}', 'k', vmin=-vlim, vmax=+vlim, cmap='RdYlBu')
    # Plot the median stamp.
    expand = (nstamps - 1) / 4
    ax = create_axis(0, nstamps - expand, expand)
    plot_stamp(Dmed[inset], None, ax, 'median')
    ax = create_axis(expand, nstamps - expand, expand)
    plot_stamp(Davg[inset], Wavg[inset], ax, 'weighted average')
    # Calculate distance matrix and plot pulls for each pair.
    dist = np.zeros((nstamps, nstamps))
    dither = np.zeros((nstamps, nstamps, 2), int)
    fscale = np.ones((nstamps, nstamps))
    for j in range(nstamps):
        D1, W1 = stamps[j]
        for i in range(j + 1, nstamps):
            D2, W2 = stamps[i]
            dist_ji, dither_ji, fscale_ji, pull = desietcimg.util.get_stamp_distance(
                D1, W1, D2, W2, maxdither=maxdither, smoothing=smoothing)
            dist[i, j] = dist[j, i] = dist_ji
            dither[j, i] = dither_ji
            dither[i, j] = -dither_ji
            fscale[j, i] = fscale_ji
            fscale[i, j] = 1 / fscale_ji
            plot_pull(pull, create_axis(i, j))
    # Plot stamps along the diagonal.
    totdist = dist.sum(axis=1)
    for j in range(nstamps):
        D, W = stamps[j]
        # Plot stamps along the diagonal.
        plot_stamp(D[inset], W[inset], create_axis(j, j), f'{totdist[j]:.2f}')
    # Find and plot the medioid stamp.
    imed = np.argmin(totdist)
    Dmedioid, Wmedioid = stamps[imed]
    ax = create_axis(0, nstamps - 2 * expand, expand)
    plot_stamp(Dmedioid[inset], Wmedioid[inset], ax, 'medioid')
    # Calculate and plot the final stack.
    ndither = 2 * maxdither + 1
    DWstack = np.zeros((ny - 2 * maxdither, nx - 2 * maxdither))
    Wstack = np.zeros_like(DWstack)
    for j in np.where(dist[imed] < maxdist)[0]:
        D, W = stamps[j]
        dy, dx = dither[imed, j]
        f = fscale[imed, j]
        inset_j = slice(maxdither + dy, ny - maxdither + dy), slice(maxdither + dx, nx - maxdither + dx)
        Dj, Wj = f * D[inset_j], W[inset_j] / f ** 2
        DWstack += Dj * Wj
        Wstack += Wj
    Dstack = np.divide(DWstack, Wstack, out=np.zeros_like(DWstack), where=Wstack > 0)
    Dstack, Wstack = desietcimg.util.normalize_stamp(Dstack, Wstack)
    ax = create_axis(expand, nstamps - 2 * expand, expand)
    plot_stamp(Dstack, Wstack, ax, 'stack')


def plot_image_quality(stacks, meta, size=33, zoom=5, pad=2, dpi=128, interpolation='none', maxline=17):
    # Calculate crops to use, without assuming which cameras are present in stacks.
    gsize, fsize = 0, 0
    for name, stack in stacks.items():
        if name.startswith('GUIDE'):
            if stack[0] is not None:
                gsize = len(stack[0])
        else:
            L, R = stack
            if L[0] is not None:
                fsize = len(L[0])
            elif R[0] is not None:
                fsize = len(R[0])
        if gsize > 0 and fsize > 0:
            break
    if gsize == 0:
        gsize = size
    if gsize == 0:
        fsize = size
    gcrop = gsize - size
    fcrop = fsize - size
    # Initialize PSF measurements.
    M = desietcimg.util.PSFMeasure(gsize)
    # Initialize the figure.
    gz = (gsize - gcrop) * zoom
    fz = (fsize - fcrop) * zoom
    nguide, nfocus = 6, 4
    width = nguide * gz + (nguide - 1) * pad
    height = gz + 2 * (fz + pad)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    # Fill the background with white.
    ax = plt.axes((0, 0, 1, 1))
    ax.axis('off')
    ax.imshow(np.ones((height, width, 3)))
    # Calculate the fiber diameter in GFA pixels.
    fiber_diam_um = 107
    pixel_size_um = 15
    #plate_scale_x, plate_scale_y = 70., 76. # microns / arcsec
    fiber_diam_pix = fiber_diam_um / pixel_size_um
    # Define helper functions.
    def imshow(ax, D, W, name):
        ax.axis('off')
        d = D.copy()
        d[W == 0] = np.nan
        vmax = np.nanpercentile(d, 99.5)
        ax.imshow(d, interpolation=interpolation, origin='lower', cmap='magma', vmin=-0.05 * vmax, vmax=vmax)
        ax.text(0, 0, name, transform=ax.transAxes, fontsize=10, color='c',
                verticalalignment='bottom', horizontalalignment='left')
        n = int(name[5])
        angle = np.deg2rad(36 * (n - 2))
        ny, nx = D.shape
        assert ny == nx
        xc, yc = 0.12 * ny, ny - 0.12 * ny
        du = 0.02 * ny * np.cos(angle)
        dv = 0.02 * ny * np.sin(angle)
        #ax.add_line(matplotlib.lines.Line2D([xc + du, xc - 3 * du], [yc - dv, yc + 3 * dv], c='c', lw=1, ls='-'))
        #ax.add_line(matplotlib.lines.Line2D([xc - dv, xc + dv], [yc - du, yc + du], c='c', lw=1, ls='-'))
        xpt = np.array([-4 * du, dv, du, -dv, -4 * du])
        ypt = np.array([4 * dv, du, -dv, -du, 4 * dv])
        ax.add_line(matplotlib.lines.Line2D(xpt + xc, ypt + yc, c='c', lw=1, ls='-'))
    # Plot GUIDEn PSFs along the middle row.
    y = (fz + pad) / height
    dy, dx = gz / height, gz / width
    fwhm_vec, ffrac_vec = [], []
    cropped = slice(gcrop // 2, gsize - gcrop // 2)
    for k, n in enumerate((2, 0, 8, 7, 5, 3)):
        x = (k * gz + (k - 1) * pad) / width
        name = 'GUIDE{0}'.format(n)
        if name in stacks:
            D, W = stacks[name]
            if D is not None:
                ax = plt.axes((x, y, dx, dy))
                xy0 = (gsize - gcrop - 1) / 2
                imshow(ax, D[cropped, cropped], W[cropped, cropped], name)
                # Draw an outline of the fiber.
                fiber = matplotlib.patches.Circle((xy0, xy0), 0.5 * fiber_diam_pix, color='c', ls='-', alpha=0.7, fill=False)
                ax.add_artist(fiber)
                # Calculate and display the PSF FWHM and fiberfrac.
                fwhm, ffrac = M.measure(D, W)
                fwhm_vec.append(fwhm if fwhm > 0 else np.nan)
                ffrac_vec.append(ffrac if ffrac > 0 else np.nan)
    # Plot FOCUSn PSFs along the top and bottom rows.
    yL = 0
    yR = (gz + 2 * pad + fz) / height
    x0 = ((fz + pad) // 2) / width
    dy, dx = fz / height, fz / width
    cropped = slice(fcrop // 2, fsize - fcrop // 2)
    for k, n in enumerate((1, 9, -1, 6, 4)):
        x = (k * gz + (k - 1) * pad) / width + x0
        if n < 0:
            xc = x
            continue
        name = 'FOCUS{0}'.format(n)
        if name in stacks:
            L, R = stacks[name]
            if L[0] is not None:
                D, W = L[0][cropped, cropped], L[1][cropped, cropped]
                ax = plt.axes((x, yL, dx, dy))
                imshow(ax, D, W, name + 'L')
            if R[0] is not None:
                D, W = R[0][cropped, cropped], R[1][cropped, cropped]
                ax = plt.axes((x, yR, dx, dy))
                imshow(ax, D, W, name + 'R')

    # Fill upper title region.
    ax = plt.axes((xc, yR, dx, dy))
    ax.axis('off')
    ax.text(0.5, 0.8, str(meta['NIGHT']), transform=ax.transAxes, fontsize=16, color='k',
            verticalalignment='bottom', horizontalalignment='center', fontweight='bold')
    ax.text(0.5, 0.6, '{0:08d}'.format(meta['EXPID']), transform=ax.transAxes, fontsize=16, color='k',
            verticalalignment='bottom', horizontalalignment='center', fontweight='bold')
    if 'PROGRAM' in meta:
        line1 = meta['PROGRAM'].strip()
        if len(line1) > maxline:
            line2 = line1[maxline:2 * maxline].strip()
            line1 = line1[:maxline].strip()
            y = 0.5
        else:
            y = 0.46
            line2 = None
        ax.text(0.5, y, line1, transform=ax.transAxes, fontsize=8, color='gray',
                verticalalignment='bottom', horizontalalignment='center')
        if line2 is not None:
            ax.text(0.5, y - 0.08, line2, transform=ax.transAxes, fontsize=8, color='gray',
                    verticalalignment='bottom', horizontalalignment='center')
    if 'MJD-OBS' in meta:
        localtime = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=meta['MJD-OBS'] - 58484.0, hours=-7)
        ax.text(0.5, 0.26, localtime.strftime('%H:%M:%S'), transform=ax.transAxes, fontsize=12, color='k',
                verticalalignment='bottom', horizontalalignment='center')
        ax.text(0.5, 0.17, 'local = UTC-7', transform=ax.transAxes, fontsize=8, color='gray',
                verticalalignment='bottom', horizontalalignment='center')
    if 'EXPTIME' in meta:
        ax.text(0.5, 0.01, '{0:.1f}s'.format(meta['EXPTIME']), transform=ax.transAxes, fontsize=12, color='k',
                verticalalignment='bottom', horizontalalignment='center')
    # Add airmass/alt, az?

    # Fill lower title region.
    ax = plt.axes((xc, yL, dx, dy))
    ax.axis('off')
    ax.text(0.5, 0.8, 'FWHM', transform=ax.transAxes, fontsize=12, color='gray',
            verticalalignment='bottom', horizontalalignment='center')
    if len(fwhm_vec) > 0:
        ax.text(0.5, 0.6, '{0:.2f}"'.format(np.nanmedian(fwhm_vec)), transform=ax.transAxes, fontsize=20, color='k',
                verticalalignment='bottom', horizontalalignment='center', fontweight='bold')
    ax.text(0.5, 0.3, 'FFRAC', transform=ax.transAxes, fontsize=12, color='gray',
            verticalalignment='bottom', horizontalalignment='center')
    if len(fwhm_vec) > 0:
        ax.text(0.5, 0.1, '{0:.0f}%'.format(100 * np.nanmedian(ffrac_vec)), transform=ax.transAxes, fontsize=20, color='k',
                verticalalignment='bottom', horizontalalignment='center', fontweight='bold')

    # Fill corner regions.
    xmirror = np.linspace(-0.8, 0.8, 15)
    ymirror = 0.1 * xmirror ** 2 - 0.85
    for k, y in enumerate((yL, yR)):
        ax = plt.axes((0, y, x0, dy))
        ax.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.plot([-0.6, 0.6], [0.8, -0.8], 'c:', lw=1)
        ax.plot([-0.6, 0.6], [-0.8, 0.8], 'c:', lw=1)
        if k == 0: # L = focus at z4 < 0 (closer to mirror)
            ax.plot([0.3, 0.6], [-0.4, -0.8], 'c-', lw=1)
            ax.plot([-0.6, -0.3], [-0.8, -0.4], 'c-', lw=1)
        else: # R = focus at z4 > 0 (closer to sky)
            ax.plot([-0.3, 0.6], [0.4, -0.8], 'c-', lw=1)
            ax.plot([-0.6, 0.3], [-0.8, 0.4], 'c-', lw=1)
        # Mirror
        ax.plot(xmirror, ymirror, 'c-', lw=3)

    hexpos = [float(Z) for Z in meta['HEXPOS'].split(',')]
    temp = meta.get('TRUSTEMP', None)
    if len(hexpos) == 6:
        ax = plt.axes((1 - x0, yR, x0, dy))
        ax.axis('off')
        ax.text(0.5, 0.85, 'hex z', transform=ax.transAxes, fontsize=10, color='k',
                verticalalignment='bottom', horizontalalignment='center')
        ax.text(0.5, 0.70, '{0:.0f}$\mu$m'.format(hexpos[2]), transform=ax.transAxes, fontsize=8, color='k',
                verticalalignment='bottom', horizontalalignment='center')
        if temp is not None:
            best = 430 + (7 - temp) * 110
            ax.text(0.5, 0.55, 'auto'.format(temp), transform=ax.transAxes, fontsize=10, color='c',
                    verticalalignment='bottom', horizontalalignment='center')
            ax.text(0.5, 0.40, '{0:.0f}$\mu$m'.format(best), transform=ax.transAxes, fontsize=8, color='c',
                    verticalalignment='bottom', horizontalalignment='center')
            ax.text(0.5, 0.25, 'truss', transform=ax.transAxes, fontsize=10, color='c',
                    verticalalignment='bottom', horizontalalignment='center')
            ax.text(0.5, 0.10, '{0:.1f}C'.format(temp), transform=ax.transAxes, fontsize=8, color='c',
                    verticalalignment='bottom', horizontalalignment='center')

    adc1, adc2 = meta.get('ADC1PHI'), meta.get('ADC2PHI')
    EL, HA, DEC = meta.get('MOUNTEL'), meta.get('MOUNTHA'), meta.get('MOUNTDEC')
    if adc1 is not None and adc2 is not None:
        axt = plt.axes((1 - x0, yL, x0, dy))
        axt.axis('off')
        axt.text(0.5, 0.9, 'ADC1 {0:.0f}$^\circ$'.format(adc1), transform=axt.transAxes, fontsize=7,
                 color='k', verticalalignment='bottom', horizontalalignment='center')
        axt.text(0.5, 0.8, 'ADC2 {0:.0f}$^\circ$'.format(adc2), transform=axt.transAxes, fontsize=7,
                 color='k', verticalalignment='bottom', horizontalalignment='center')
        if EL is not None:
            axt.text(0.5, 0.7, 'ELEV {0:.0f}$^\circ$'.format(EL), transform=axt.transAxes, fontsize=7,
                     color='gray', verticalalignment='bottom', horizontalalignment='center')
        ax = plt.axes((1 - x0, yL, x0, x0 * width / height))
        ax.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        r = 0.98
        circle = matplotlib.patches.Circle((0, 0), r, color='lightgray', ls='-', fill=False)
        ax.add_artist(circle)
        # Draw the horizon and zenith.
        ax.plot([-r, r], [0, 0], '-', c='lightgray', lw=1)
        ax.plot([0, 0], [0, r], '-', c='lightgray', lw=1)
        # Draw the actual ADC angles.
        for phi in adc1, adc2:
            phi = np.deg2rad(phi)
            ax.plot([0, r * np.sin(phi)], [0, r * np.cos(phi)], 'k-', lw=1)
        if not (EL is None or HA is None or DEC is None):
            # Following slide 20 of DESI-3522.
            PARA, PHI1, PHI2 = desietcimg.util.ADCangles(EL, HA, DEC)
            axt.text(0.5, 0.6, 'PARA {0:.0f}$^\circ$'.format(PARA), transform=axt.transAxes, fontsize=7,
                     color='c', verticalalignment='bottom', horizontalalignment='center')
            PARA, PHI1, PHI2, EL = np.deg2rad([PARA, PHI1, PHI2, EL])
            # Draw the ADC angles necessary to cancel atmospheric refraction at HA.
            HORIZON = PARA + 0.5 * np.pi
            for phi in PHI1, PHI2:
                u, v = r * np.cos(phi - HORIZON), r * np.sin(phi - HORIZON)
                ax.plot([-u, u], [v, -v], ':', c='gray', lw=1)
            # Draw the elevation angle.
            u, v = r * np.cos(EL), r * np.sin(EL)
            ax.plot([-u, 0, u], [v, 0, v], '-', c='gray', lw=1)
            # Draw a North pointer at the parallactic angle relative to zenith.
            u, v = 0.1 * r * np.sin(PARA), 0.1 * r * np.cos(PARA)
            ax.plot([-u, v, 6 * u, -v, -u], [-v, -u, 6 * v, u, -v], 'c-', lw=2)

    return fig


def plot_guide_stars(Dsum, WDsum, Msum, params, night, expid, camera, maxdxy=5):
    """Plot guide star analysis results.
    """
    fig, ax = plt.subplots(5, 1,  figsize=(12, 15), sharex=True)
    nstars, nframes = params.shape[:2]
    t = np.arange(nframes)
    # Conversion from pix to mas.
    conv = 1e3 * 15 / 70.54
    for P in params:
        x, y = P[:, 0], P[:, 1]
        x0, y0 = np.median(x), np.median(y)
        # Fit and plot straight lines to model the trend.
        xsel = np.abs(x - x0) < maxdxy
        ysel = np.abs(y - y0) < maxdxy
        p1x, p0x = np.polyfit(t[xsel], x[xsel], deg=1)
        p1y, p0y = np.polyfit(t[ysel], y[ysel], deg=1)
        # Plot relative to the median values.
        xfit = p0x + t * p1x
        yfit = p0y + t * p1y
        line2d = ax[0].plot(t, xfit, '-', lw=2, alpha=0.5)
        c=line2d[0].get_color()
        ax[1].plot(t, yfit, '-', lw=2, alpha=0.5, c=c)
        # Calculate std dev relative to the linear trend converted to mas.
        xstd = conv * np.std(x[xsel] - xfit[xsel])
        ystd = conv * np.std(y[xsel] - yfit[xsel])
        # Plot per-frame centroids labeled with std dev.
        ax[0].plot(t, x, '-', c=c, alpha=0.5)
        ax[1].plot(t, y, '-', c=c, alpha=0.5)
        ax[0].plot(t, x, '.', c=c, label='std={0:.1f} mas'.format(xstd))
        ax[1].plot(t, y, '.', c=c, label='std={0:.1f} mas'.format(ystd))
        ax[2].plot(P[:, 2], c=c)
        ax[3].plot(P[:, 3], c=c)
        ax[4].plot(P[:, 4], c=c)
    ax[0].legend(ncol=nstars)
    ax[1].legend(ncol=nstars)
    ax[0].set_ylim(-maxdxy, +maxdxy)
    ax[1].set_ylim(-maxdxy, +maxdxy)
    ax[0].set_ylabel('dX [pix]', fontsize=12)
    ax[1].set_ylabel('dY [pix]', fontsize=12)
    ax[2].set_ylabel('Transparency', fontsize=12)
    ax[3].set_ylabel('Fiber Fraction', fontsize=12)
    ax[4].set_ylabel('Fit Min NLL', fontsize=12)
    ax[2].set_ylim(-0.1, 1.1)
    ax[3].set_ylim(-0.1, 1.1)
    ax[4].set_yscale('log')
    ax[4].set_ylim(0.1, 100)
    ax[4].set_xlabel('{0} {1} {2} Frame #'.format(night, expid, camera), fontsize=14)
    ax[4].set_xlim(-1.5, nframes + 0.5)
    plt.subplots_adjust(0.07, 0.04, 0.99, 0.99, hspace=0.03)
    return fig, ax
