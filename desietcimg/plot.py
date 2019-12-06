"""Plotting utilities. Import requires matplotlib.
"""
import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
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
    cmap = matplotlib.cm.get_cmap(args['cmap'])
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
        ax.text(0.01, 0.01 * nx / ny, label, transform=ax.transAxes, **args)
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
    print('clip', clip_vmin, clip_vmax, vmin, vmax)
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
    cmap = matplotlib.cm.get_cmap(cmap)
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
    cmap = matplotlib.cm.get_cmap(cmap)
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
    cmap = matplotlib.cm.get_cmap(cmap)
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
