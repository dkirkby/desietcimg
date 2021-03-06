"""General purpose utilities for imaging analysis.
"""
import os
import re
import pathlib
import logging

import numpy as np

import scipy.signal
import scipy.stats
import scipy.ndimage

import fitsio


def downsample(data, downsampling, summary=np.sum, allow_trim=False):
    """Downsample a 2D array.

    Parameters
    ----------
    data : array
        Two dimensional array of values to downsample.
    downsampling : int
        Downsampling factor to use along both dimensions. Must evenly divide the
        data dimensions when allow_trim is False.
    summary : callable
        The summary function to use that will be applied to each block of shape
        (dowsampling, downsampling) to obtain the output downsampled values.
        Must support broadcasting and an axis parameter. Useful choices are
        np.sum, np.mean, np.min, np.max, np.median, np.var but any ufunc
        should work.
    allow_trim : bool
        When False, the input dimensions (ny, nx) must both exactly divide
        the downsampling value.  Otherwise, any extra rows and columns are
        silently trimmed before apply the summary function.

    Returns
    -------
    array
        A two dimensional array of shape (ny // downsampling, nx // downsampling)
        where the input data shape is (ny, nx).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError('Data must be 2 dimensional.')
    ny, nx = data.shape
    if not allow_trim and ((nx % downsampling) or (ny % downsampling)):
        raise ValueError('Data shape {0} does not evenly divide downsampling={1} and allow_trim is False.'
                         .format((ny, nx), downsampling))
    ny //= downsampling
    nx //= downsampling
    shape = (ny, nx, downsampling, downsampling)
    strides = (downsampling * data.strides[0], downsampling * data.strides[1]) + data.strides
    blocks = np.lib.stride_tricks.as_strided(
        data[:downsampling * ny, :downsampling * nx], shape=shape, strides=strides)
    return summary(blocks, axis=(2, 3))


def downsample_weighted(D, W, downsampling=4, allow_trim=True):
    """Downsample 2D data D with weights W.
    """
    if D.shape != W.shape:
        raise ValueError('Arrays D, W must have the same shape.')
    if D.ndim != 2:
        raise ValueError('Arrays D, W must be 2D.')
    if np.any(W < 0):
        raise ValueError('Array W contains negative values.')
    WD = downsample(D * W, downsampling=downsampling, summary=np.sum, allow_trim=allow_trim)
    W = downsample(W, downsampling=downsampling, summary=np.sum, allow_trim=allow_trim)
    D = np.divide(WD, W, out=np.zeros_like(WD), where=W > 0)
    return D, W


def preprocess(D, W, nsig_lo=10, nsig_hi=30, vmin=None, vmax=None):
    """Preprocess weighted 2D array data for display.
    """
    masked = W == 0
    # Calculate the median unmasked pixel value.
    median_value = np.median(D[~masked])
    # Calculate the median non-zero inverse variance.
    median_ivar = np.median(W[~masked])
    # Calculate the corresponding pixel sigma.
    sigma = 1 / np.sqrt(median_ivar)
    if vmin is None:
        vmin = median_value - nsig_lo * sigma
    if vmax is None:
        vmax = median_value + nsig_hi * sigma
    # Clip values to [vmin, vmax].
    D = np.clip(D, vmin, vmax)
    # Set masked pixel values to nan so they are not plotted.
    D[masked] = np.nan
    return D


def smooth(D, W, smoothing):
    """Apply a weighted Gaussian smoothing.
    """
    WD = scipy.ndimage.gaussian_filter(W * D, smoothing)
    W = scipy.ndimage.gaussian_filter(W, smoothing)
    D = np.divide(WD, W, out=np.zeros_like(D), where=W > 0)
    return D, W


def get_significance(D, W, smoothing=2.5, downsampling=2, medfiltsize=5):
    """Calculate a downsampled pixel significance image.

    This function is a quick and robust way to calculate a significance
    image suitable for thresholding to identify regions likely to contain
    a source.  There are three stages:

      - Apply a weighted Gaussian filter,
      - Perform a weighted downsampling,
      - Estimate and subtract a background image using a median filter.

    This is designed to work with :func:`detect_sources`.

    Parameters
    ----------
    D : array
        2D array of pixel values.
    W : array
        2D array of corresponding ivar weights with same shape as D.
    smoothing : float
        Gaussian smoothing sigma to apply in pixels before downsampling.
    downsampling : int
        Downsampling factor to apply. Must evenly divide both dimensions of D.
    medfiltsize : int
        Size of median filter to apply after after downsampling to estimate
        the smoothly varying background. Must be odd.

    Returns
    -------
    array
        2D array of downsampled pixel significance values. Note that the
        returned dimensions are downsampled relative to the input arrays.
    """
    # Apply weighted smoothing.
    D, W = smooth(D, W, smoothing)
    # Downsample.
    D, W = downsample_weighted(D, W, downsampling=downsampling, allow_trim=False)
    # Median filter the data to estimate background variations.
    mask = W == 0
    D[mask] = np.median(D[~mask])
    Dm = scipy.ndimage.median_filter(D, medfiltsize)
    # Subtract the median-filtered image.
    D -= Dm
    # Estimate the significance of each (downsampled) pixel.
    return D * np.sqrt(W)


def detect_sources(snr, minsnr=4, minsize=8, maxsize=32, minsep=0,
                   min_snr_ratio=0.1, maxsrc=20, measure=None):
    """Detect and measure sources in a significance image.

    A source is defined as a connected and isolated region of pixels above
    some threshold that fits within a square bounding box with a size in
    the range ``minsize`` to ``maxsize`` pixels.

    When ``measure`` is None, the ``maxsrc`` sources with the
    highest total SNR are returned with their total SNR and centroid
    coordinates measured.  When a callable ``measure`` is supplied, it
    is passed the total SNR and centroid coordinates and can either return
    None to reject a source, or return an updated set of measurements.

    Parameters
    ----------
    snr : array
        2D image of pixel significances, e.g., from :func:`get_significance`.
    minsnr : float
        All pixels above this threshold will be assigned to a potential
        source.
    minsize : int
        Minimum square bounding box size for a source.
    maxsize : int
        Maximum square bounding box size for a source.
    minsep : float
        Minimum distance between any pair of detected sources in pixels.
        Distances are measured between the SNR ** 2 weighted centers of
        gravity of each source candidate.
    maxsrc : int
        Maximum number of measured sources to return.
    measure : callable or None
        Optional function that is passed the total SNR and centroid coordinates
        of a candidate source and either returns None to reject the source or
        an updated set of measurements.

    Returns
    -------
    list
        A list of the measurements for each detected source.
    """
    if minsize > maxsize:
        raise ValueError('Expected minsize <= maxsize.')
    ny, nx = snr.shape
    # Label all non-overlapping regions above SNRmin in the inset image.
    labeled, nlabels = scipy.ndimage.label(snr > minsnr)
    if nlabels == 0:
        return []
    labels = np.arange(1, nlabels + 1)
    # Calculate bounding boxes for each candidate source.
    bboxes = scipy.ndimage.find_objects(labeled)
    # Estimate the quadrature summed SNR for each candidate source.
    snrtot = scipy.ndimage.labeled_comprehension(
        snr, labeled, labels, out_dtype=float, default=-1,
        func=lambda X: np.sqrt(np.sum(X ** 2)))
    maxsnrtot = None
    # Rank sources by snrtot.
    ranks = np.argsort(snrtot)[::-1]
    # Build the final list of detected sources.
    sources = []
    snrsq = snr ** 2
    minsepsq = minsep ** 2
    centroids = np.empty((maxsrc, 2))
    for idx in range(nlabels):
        label = labels[ranks[idx]]
        srcsnrtot = snrtot[label - 1]
        if maxsnrtot is not None and srcsnrtot < min_snr_ratio * maxsnrtot:
            break
        # Lookup this source's bounding box.
        yslice, xslice = bboxes[label - 1]
        size = max(yslice.stop - yslice.start, xslice.stop - xslice.start)
        if size < minsize or size > maxsize:
            continue
        # Calculate the SNR**2 weighted center of mass for this source.
        yc, xc = scipy.ndimage.center_of_mass(snrsq, labeled, label)
        nsrc = len(sources)
        if nsrc > 0 and minsep > 0:
            # Calculate the distance to each previous source.
            rsq = np.sum((centroids[:nsrc] - np.array([xc, yc])) ** 2, axis=1)
            if np.any(rsq < minsepsq):
                continue
        params = (srcsnrtot, xc, yc, yslice, xslice)
        if measure is not None:
            params = measure(*params)
            if params is None:
                continue
        centroids[nsrc] = (xc, yc)
        if maxsnrtot is None:
            maxsnrtot = srcsnrtot
        sources.append(params)
        if len(sources) == maxsrc:
            break
    return sources


def estimate_bg(D, W, margin=4, maxchisq=2, minbgfrac=0.2):
    """Estimate the background level from the margins of an image.

    Parameters
    ----------
    D : array
        2D array of pixel values.
    W : array
        2D array of corresponding inverse variances.
    margin : int
        Size of margin around the outside of the image to use to
        estimate the background.
    maxchisq : float
        Maximum pixel chi-square value to consider a margin
        pixel as background like.
    minbgfrac : float
        Minimum fraction of background-like margin pixels required
        to use a weighted mean value estimate.  Otherwise, a
        noisier but more robust median of unmasked margin pixel
        values is returned.

    Returns
    -------
    float
        Estimate of the background level. Will be zero if all
        margin pixels are masked.
    """
    mask = np.zeros(D.shape, bool)
    mask[:margin] = mask[-margin:] = True
    mask[:, :margin] = mask[:, -margin:] = True
    # Find the median unmasked pixel value in the margin.
    d = D[margin]
    w = W[margin]
    if not np.any(w > 0):
        # There are no unmasked margin pixels.
        return 0
    med = np.median(d[w > 0])
    # Find the median unmasked ivar in the margin.
    sig = 1 / np.sqrt(np.median(w[w > 0]))
    # Select bg-like pixels in the margin.
    chisq = w * (d - med) ** 2
    bg = (chisq < maxchisq) & (w > 0)
    if np.count_nonzero(bg) < minbgfrac * d.size:
        # Return the median when there are not enough bg pixels.
        return med
    else:
        # Calculate a weighted mean of the bg pixels.
        return np.sum(w[bg] * d[bg]) / np.sum(w[bg])


def normalize_stamp(D, W, smoothing=2.5):
    """Normalize a stamp to its weighted mean value.
    Should generally subtract a background estimate first.
    """
    smoothed, _ = smooth(D, W, smoothing)
    norm = smoothed.sum()
    if norm != 0:
        return D / np.abs(norm), W * norm ** 2


def get_stamp_distance(D1, W1, D2, W2, maxdither=3, smoothing=1, fscale=np.linspace(0.85, 1.15, 11)):
    """Calculate the minimum chisq distance between two stamps allowing for some dither.
    """
    ny, nx = D1.shape
    assert D1.shape == D2.shape == W1.shape == W2.shape
    nscale = len(fscale)
    fvec = fscale.reshape(-1, 1, 1)
    # Smooth both stamps.
    D1, W1 = smooth(D1, W1, smoothing)
    D2, W2 = smooth(D2, W2, smoothing)
    # Inset the first stamp by the dither size.
    inset = slice(maxdither, ny - maxdither), slice(maxdither, nx - maxdither)
    D1inset = D1[inset]
    W1inset = W1[inset]
    # Loop over dithers of the second stamp.
    ndither = 2 * maxdither + 1
    pull = np.zeros((ndither, ndither, nscale, ny - 2 * maxdither, nx - 2 * maxdither))
    dxy = np.arange(-maxdither, maxdither + 1)
    for iy, dy in enumerate(dxy):
        for ix, dx in enumerate(dxy):
            # Dither the second stamp.
            D2inset = D2[maxdither + dy:ny - maxdither + dy, maxdither + dx:nx - maxdither + dx]
            W2inset = W2[maxdither + dy:ny - maxdither + dy, maxdither + dx:nx - maxdither + dx]
            # Calculate the chi-square distance between the inset stamps with scale factors of
            # 1/fvec and fvec applied to (D1,W1) and (D2,W2) respectively.
            num = np.sqrt(W1inset * W2inset) * (D1inset / fvec - D2inset * fvec)
            denom = np.sqrt(W1inset * fvec ** 2 + W2inset * fvec ** -2)
            # Could also use where=(num > 0) here.
            pull[iy, ix] = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    # Find the dither with the smallest chisq.
    chisq = np.sum(pull ** 2, axis=(3, 4))
    iy, ix, iscale = np.unravel_index(np.argmin(chisq.reshape(-1)), (ndither, ndither, nscale))
    assert chisq.min() == chisq[iy, ix, iscale]
    # Return the smallest distance, the corresponding dither and scale, and the best pull image.
    return (chisq[iy, ix, iscale] / D1inset.size, np.array((dxy[iy], dxy[ix]), int),
            fscale[iscale], pull[iy, ix, iscale].copy())


def get_stacked(stamps, smoothing=1, maxdither=1, maxdist=3, min_stack=2):
    """Calculate a stack of detected sources ignoring outliers.
    """
    # Extract and normalize stamps.
    nstamps = len(stamps)
    if nstamps == 0:
        return None, None
    stamps = [normalize_stamp(*S[2:4]) for S in stamps]
    ny, nx = (stamps[0][0]).shape
    # Calculate distance matrix and record best dithers and scales.
    dist = np.zeros((nstamps, nstamps))
    dither = np.zeros((nstamps, nstamps, 2), int)
    fscale = np.ones((nstamps, nstamps))
    for j in range(nstamps):
        D1, W1 = stamps[j]
        for i in range(j + 1, nstamps):
            D2, W2 = stamps[i]
            dist_ji, dither_ji, fscale_ji, _ = get_stamp_distance(
                D1, W1, D2, W2, maxdither=maxdither, smoothing=smoothing)
            dist[i, j] = dist[j, i] = dist_ji
            dither[j, i] = dither_ji
            dither[i, j] = -dither_ji
            fscale[j, i] = fscale_ji
            fscale[i, j] = 1 / fscale_ji
    # Find the medioid stamp.
    totdist = dist.sum(axis=1)
    imed = np.argmin(totdist)
    # How many other stamps are close enough to stack?
    stack_idx = np.where(dist[imed] < maxdist)[0]
    if len(stack_idx) < min_stack:
        # Calculate and return the weighted average stamp.
        DWsum = np.sum(np.stack([D * W for D, W in stamps]), axis=0)
        Wavg = np.sum(np.stack([W for D, W in stamps]), axis=0)
        Davg = np.divide(DWsum, Wavg, out=np.zeros_like(DWsum), where=Wavg > 0)
        inset = slice(maxdither, ny - maxdither), slice(maxdither, nx - maxdither)
        return normalize_stamp(Davg[inset], Wavg[inset])
    # Calculate the final stack.
    ndither = 2 * maxdither + 1
    DWstack = np.zeros((ny - 2 * maxdither, nx - 2 * maxdither))
    Wstack = np.zeros_like(DWstack)
    for j in stack_idx:
        D, W = stamps[j]
        dy, dx = dither[imed, j]
        f = fscale[imed, j]
        inset_j = slice(maxdither + dy, ny - maxdither + dy), slice(maxdither + dx, nx - maxdither + dx)
        Dj, Wj = f * D[inset_j], W[inset_j] / f ** 2
        DWstack += Dj * Wj
        Wstack += Wj
    Dstack = np.divide(DWstack, Wstack, out=np.zeros_like(DWstack), where=Wstack > 0)
    return normalize_stamp(Dstack, Wstack)


def make_template(size, profile, dx=0, dy=0, oversampling=10, normalized=True):
    """Build a square template for an arbitrary profile.

    Parameters
    ----------
    size : int
        Output 2D array will have shape (size, size).
    profile : callable
        Function of (x,y) that evaluates the profile to use, where x and y are arrays
        of pixel coordinates relative to the template center. This function is called
        once, instead of iterating over pixels, so should broadcast over x and y.
    dx : float
        Offset values of x passed to the profile by this amount (in pixels).
    dy : float
        Offset values of y passed to the profile by this amount (in pixels).
    oversampling : int
        Integrate over the template pixels by working on a finer grid with this
        oversampling factor, then downsample to the output pixels.
    normalized : bool
        When True, the sum of output pixels is normalized to one.

    Returns
    -------
    array
        2D numpy array of template pixel values with shape (size, size).
    """
    xy = (np.arange(size * oversampling) - 0.5 * (size * oversampling - 1)) / oversampling
    z = profile(xy - dx, (xy - dy).reshape(-1, 1))
    T = downsample(z, oversampling, np.mean)
    if normalized:
        T /= T.sum()
    return T


def fiber_profile(x, y, r0, blur=0.1):
    """Radial profile of a blurred disk.

    This implementation approximates a 2D Gaussian blur using a 1D erf,
    so is only exact in the limit of zero blur (because the 2D
    Jacobian requires less blur to r > r0 than r < r0 to preserve area).
    This approximation means that we are assuming a slightly assymetric
    blur.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 + 0.5 * scipy.special.erf((r0 - r) / (np.sqrt(2) * blur))


class Convolutions(object):
    """Convolution of one or more sources with the same kernel.

    Changes to :attr:`sources` using :meth:`set_source` will automatically
    update :attr:`convolved` be only recomputing convolutions in the
    region affected by the change.

    The convolutions uses the `same` mode.
    """
    def __init__(self, sources, kernel):
        ksize = len(kernel)
        if kernel.shape != (ksize, ksize) or ksize % 2 != 1:
            raise ValueError('Kernel must be 2D and square with odd size.')
        self.kernel = np.asarray(kernel)
        ny, nx = sources[0].shape
        for source in sources[1:]:
            if source.shape != (ny, nx):
                raise ValueError('All sources must have the same shape.')
        self.sources = [np.asarray(source) for source in sources]
        self.convolved = [
            scipy.signal.convolve(source, kernel, mode='same')
            for source in sources]

    def set_source(self, yslice, xslice, value):
        if not isinstance(yslice, slice):
            yslice = slice(yslice, yslice + 1)
        if not isinstance(xslice, slice):
            xslice = slice(xslice, xslice + 1)
        ny, nx = self.sources[0].shape
        h = len(self.kernel) // 2
        # Calculate the region of the convolution affected by this change.
        xloc = max(0, xslice.start - h)
        yloc = max(0, yslice.start - h)
        xhic = min(nx, xslice.stop + h)
        yhic = min(ny, yslice.stop + h)
        conv_slice = (slice(yloc, yhic), slice(xloc, xhic))
        # Calculate the region of the source that that contributes to the change.
        xlos = max(0, xloc - h)
        ylos = max(0, yloc - h)
        xhis = min(nx, xhic + h)
        yhis = min(ny, yhic + h)
        assert xlos <= xloc and ylos <= yloc and xhis >= xhic and yhis >= yhic
        # Calculate the slice of the reconvolved output to use.
        x1 = xloc - xlos
        y1 = yloc - ylos
        x2 = x1 + xhic - xloc
        y2 = y1 + yhic - yloc
        for source, convolved in zip(self.sources, self.convolved):
            source[yslice, xslice] = value
            # Reconvolve the changed region.
            reconv = scipy.signal.convolve(
                source[ylos:yhis, xlos:xhis], self.kernel, mode='same')
            # Copy the changed region into the full convolution.
            convolved[conv_slice] = reconv[y1:y2, x1:x2]
        return conv_slice


def prepare(D, W=None, invgain=1.6, smoothing=3, saturation=None):
    """Prepare image data for analysis.

    The input data D is converted to float32, if necessary, and an
    estimate of the mean background will be subtracted.

    If no inverse variance map W is provided, it will be estimated
    from D, including both background and signal contributions.
    Otherwise, just return W converted to float32.

    Parameters
    ----------
    D : array
        2D array of pixel values in ADU units. An estimate of the
        mean background will be subtracted.
    W : array or None
        2D array of inverse variance weights in ADU units, or None
        if this should be estimated from D.
    invgain : float
        Inverse gain in units of e/ADU to assume for estimating
        the signal variance contribution.
    smoothing : int
        Number of pixels for median filtering of D used to estimate
        the signal variance contribution. Must be odd.
    saturation : int or None
        Pixel values >= this level are considered saturated and masked.
        Nothing is maksed when saturation is None.

    Returns
    -------
    tuple
        Tuple D, W of 2D numpy float32 arrays.
    """
    # Default saturation level is the maximum value for this datatype.
    if saturation is None:
        saturated = np.zeros(D.shape, bool)
    else:
        saturated = (D >= saturation)
        logging.info('Found {np.count_nonzero(saturated)} pixels saturated (>={0}).'
                     .format(saturation))
    # Convert to a float32 array.
    D = np.array(D, np.float32)
    # Select background pixels using sigma clipping.
    clipped, _, _ = scipy.stats.sigmaclip(D[~saturated])
    # Subtract the clipped mean from the data.
    bgmean = np.mean(clipped)
    logging.info('Subtracted background mean {0:.1f} ADU.'.format(bgmean))
    D -= bgmean
    if W is None:
        # Use the clipped pixels for the background variance estimate.
        bgvar = np.var(clipped)
        logging.info('Estimated background RMS {0:.1f} ADU.'.format(np.sqrt(bgvar)))
        var = bgvar * np.ones_like(D)
        # Estimate additional variance due to median-filtered signal.
        Dsmoothed = scipy.signal.medfilt2d(D, smoothing)
        #smoother = np.ones((smoothing, smoothing)) / smoothing ** 2
        #Dsmoothed = scipy.signal.convolve(D, smoother, mode='same')
        var += np.maximum(0., Dsmoothed) / invgain
        # Build an inverse variance image with zeros where var is zero.
        W = np.divide(1., var, out=np.zeros_like(var, dtype=np.float32), where=var > 0)
    else:
        W = np.asarray(W, dtype=np.float32)
    # Zero ivar for any saturated pixels.
    W[saturated] = 0.
    return D, W


def sobelfilter(D, W):
    """Estimate the magnitude of the 2D gradient of D.

    Uses Sobel filters in x and y modified to account for ivar weights W.
    """
    here, plus, minus = slice(1, -1), slice(2, None), slice(None, -2)
    # Estimate slopes along each axis at each pixel.
    Dx = 0.5 * (D[:, plus] - D[:, minus])
    Dy = 0.5 * (D[plus, :] - D[minus, :])
    #  Calculate  the corresponding inverse variances.
    Wp, Wm = W[:, plus], W[:, minus]
    Wx = 0.25 * np.divide(Wp * Wm, Wp + Wm, out=np.zeros_like(Wp), where=Wp + Wm > 0)
    Wp, Wm = W[plus, :], W[minus, :]
    Wy = 0.25 * np.divide(Wp * Wm, Wp + Wm, out=np.zeros_like(Wp), where=Wp + Wm > 0)
    # Average slope estimates along the other axis with weights (1, 2, 1).
    WDx = Wx[minus, :] * Dx[minus, :] + 2 * Wx[here, :] * Dx[here, :] + Wx[plus, :] * Dx[plus, :]
    Wxsum = Wx[minus, :] + 2 * Wx[here, :] + Wx[plus, :]
    Dx = np.divide(WDx, Wxsum, out=np.zeros_like(WDx), where=Wxsum > 0)
    WDy = Wy[:, minus] * Dy[:, minus] + 2 * Wy[:, here] * Dy[:, here] + Wy[:, plus] * Dy[:, plus]
    Wysum = Wy[:, minus] + 2 * Wy[:, here] + Wy[:, plus]
    Dy = np.divide(WDy, Wysum, out=np.zeros_like(WDy), where=Wysum > 0)
    # Estimate the 2D gradient magnitude.
    Dg = np.zeros_like(D)
    Dg[here, here] = np.hypot(Dx, Dy)
    return Dg


def mask_defects(D, W, chisq_max=5e3, kernel_size=3, min_neighbors=7, inplace=False):
    if not inplace:
        W = W.copy()
    # Initialize the kernel.
    if kernel_size % 2 == 0:
        raise ValueError('Kernel size must be odd.')
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    nby2 = kernel_size // 2
    max_neighbors = kernel_size ** 2 - 1
    kernel[nby2, nby2] = 0.
    # Calculate the ivar-weighted image.
    WD = np.array(W * D, np.float32)
    # Convolve with the kernel.
    C = Convolutions([WD, W], kernel)
    WD, W = C.sources
    WDf, Wf = C.convolved
    # Calculate the Wf weighted residuals.
    res = Wf * D - WDf
    # Calculate residual chisq.
    denom = (W + Wf) * Wf
    Wratio = np.divide(W, denom, out=np.zeros_like(W), where=denom != 0)
    chisq = res ** 2 * Wratio
    # Iteratively remove defects.
    nmasked = 0
    ny, nx = D.shape
    while np.any(chisq > chisq_max):
        # Find the next largest chisq.
        iy, ix = np.unravel_index(np.argmax(chisq), (ny, nx))
        # Count the number of surrounding pixels with nonzero ivar.
        xlo, ylo = max(0, ix - nby2), max(0, iy - nby2)
        xhi, yhi = min(nx, ix + nby2 + 1), min(ny, iy + nby2 + 1)
        # Subtract 1 since chisq > 0 means that W > 0 for the central pixel.
        num_neighbors = np.count_nonzero(W[ylo:yhi, xlo:xhi]) - 1
        if num_neighbors < min_neighbors or ((num_neighbors < max_neighbors) and (chisq[iy, ix] < 2 * chisq_max)):
            # Zero this pixel's chisq without changing its weight.
            chisq[iy, ix] = 0.
            continue
        # Set this pixel's ivar to zero.
        changed = C.set_source(iy, ix, 0)
        # Update the chisq.
        res[changed] = Wf[changed] * D[changed] - WDf[changed]
        denom = (W[changed] + Wf[changed]) * Wf[changed]
        Wratio[changed] = 0
        np.divide(
            W[changed], denom, out=Wratio[changed], where=denom != 0)
        chisq[changed] = res[changed] ** 2 * Wratio[changed]
        nmasked += 1
    return W, nmasked


def get_data(name, must_exist=False):
    """Return the absolute path to a named data file associated with this package.

    Relative paths refer to the desietcimg/data/ folder of this installation.
    Use an absolute path to override this behavior.
    """
    if os.path.isabs(name):
        path = name
    else:
        import desietcimg
        root = os.path.abspath(os.path.dirname(desietcimg.__file__))
        path = os.path.join(root, 'data', name)
    if must_exist and not os.path.exists(path):
        raise RuntimeError('Non-existent data file: {0}'.format(path))
    return path


def find_files(pattern, min=None, max=None, check_parent=True, partial_match_is_error=True):
    """Find files matching a pattern with a sequence number.

    The sequence number is represented using {N} in the input pattern,
    which can be repeated.

    Parameters
    ----------
    pattern : str or pathlib.Path
        File pattern using {N} to represent the sequence number.
    min : int or None
        Only values of N >= min will be returned.
    max : int or None
        Only values of N <= max will be returned.
    check_parent : bool
        Raise an exception when True and the parent directory does
        not exist.
    partial_match_is_error : bool
        Raise an exception when True for any paths that match the
        first {N} but not all subsequent {N}'s in the input pattern.

    Returns
    -------
    list
        List of filenames matching the pattern and filtered by
        any min/max cuts.
    """
    if not isinstance(pattern, pathlib.Path):
        pattern = pathlib.Path(pattern)
    # Find which parts of the pattern contain {N}.
    parts = pattern.parts
    part_has_N = ['{N}' in part for part in parts]
    if not any(part_has_N):
        # Nothing to match. Return the input if it exists.
        return [str(pattern)] if pattern.exists() else []
    first_N = part_has_N.index(True)
    # Build the parent path to search.
    parent_path = pathlib.Path(*parts[:first_N])
    if check_parent and not parent_path.exists():
        raise FileNotFoundError(parent_path)
    # Build the suffix patttern if there is one.
    remaining = first_N + 1
    suffix_pattern = str(pathlib.Path(*parts[remaining:])) if remaining < len(parts) else None
    # Look for paths matching the first {N} in the path using * as a glob pattern.
    first_N_pattern = parts[first_N]
    paths = sorted([str(P) for P in parent_path.glob(first_N_pattern.format(N='*'))])
    # Check for integer matches to N.
    regexp = re.compile(first_N_pattern.format(N='([0-9]+)') + '$')
    selected = []
    suffix = ''
    for path in paths:
        found = regexp.search(path)
        if found:
            N = int(found.group(1))
            if min is not None and N < min:
                continue
            if max is not None and N > max:
                continue
            if suffix_pattern:
                # Build the full path for this value of N.
                # Use the regexp string match rather than the integer N to preserve formatting.
                suffix = suffix_pattern.format(N=found.group(1))
                full_path = pathlib.Path(path) / suffix
                # Silently ignore paths that match the first {N} but not subsequent ones.
                if not full_path.exists():
                    if partial_match_is_error:
                        raise ValueError(
                            'Partial match error: found {path} but not {full_path}.'
                            .format(path=path, full_path=full_path))
                    else:
                        continue
                path = str(path)
            selected.append(str(pathlib.Path(path) / suffix))
    return selected


def load_raw(files, *keys, hdu=0, slices=None):
    """ Load a sequence of raw data from FITS files into a single array.

    Parameters
    ----------
    files : iterable or str
        List of filenames to read or a pattern that will be passed to
        :func:`find_files`.
    keys : variable args
        Header keywords to read from each file.
    hdu : int or str
        Index or name of the HDU containing the raw data and header keywords.
    slices : tuple or None
        Only load the specified tuple of slices or load the full image when None.

    Returns
    -------
    tuple
        Tuple (raw, meta) where raw is a numpy array of shape (nexp, ny, nx)
        and the specified dtype, containing the contents of each file in the
        order they are listed in the input files, and meta is a dictionary
        of arrays containing the specified header values (or None when a key
        is not present in the header).
    """
    if isinstance(files, (str, pathlib.Path)):
        files = find_files(files)
    nexp = len(files)
    if nexp == 0:
        logging.warning('No files specified.')
        return None, None
    if slices is None:
        slices = (slice(None), slice(None))
    meta = {key: [] for key in keys}
    for k, file in enumerate(files):
        with fitsio.FITS(file, mode='r') as hdus:
            if k == 0:
                data = hdus[hdu][slices]
                raw = np.empty((nexp,) + data.shape, data.dtype)
                raw[0] = data
                logging.debug('Reading {0} files with shape {1} and dtype {2}.'
                              .format(nexp, data.shape, data.dtype))
            else:
                raw[k] = hdus[hdu][slices]
            hdr = hdus[hdu].read_header()
            for key in keys:
                meta[key].append(hdr.get(key, None))
    # Convert each list of metadata values to a numpy array.
    for key in meta:
        meta[key] = np.array(meta[key])
    return raw, meta


class PSFMeasure(object):

    def __init__(self, stamp_size, fiber_diam_um=107, pixel_size_um=15, plate_scales=(70., 76.),
                 max_offset_pix=3.5, noffset=15, nangbins=20):
        self.stamp_size = stamp_size
        self.pixel_size_um = pixel_size_um
        self.plate_scales = plate_scales
        # Tabulate fiber templates for each (x,y) offset in the x >= 0 and y >= 0 quadrant.
        self.offset_template = np.empty((noffset, noffset, stamp_size, stamp_size), np.float32)
        max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
        profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < max_rsq)
        delta = np.linspace(0, max_offset_pix, noffset)
        for iy in range(noffset):
            for ix in range(noffset):
                self.offset_template[iy, ix] = make_template(
                    stamp_size, profile, dx=delta[ix], dy=delta[iy], normalized=False)
        self.xyoffset = np.linspace(-max_offset_pix, +max_offset_pix, 2 * noffset - 1)
        dxy = np.arange(self.stamp_size) - 0.5 * (self.stamp_size - 1)
        self.xgrid, self.ygrid = np.meshgrid(dxy, dxy, sparse=False)
        rmax = dxy[-1] * self.pixel_size_um / max(self.plate_scales)
        self.angbins = np.linspace(0., rmax, nangbins + 1)
        self.rang = 0.5 * (self.angbins[1:] + self.angbins[:-1])

    def measure(self, P, W):
        assert P.shape == W.shape == (self.stamp_size, self.stamp_size)
        Psum = np.sum(P)
        # Prepare the array of fiber fractions for offsets in all 4 quadrants.
        nquad = len(self.offset_template)
        nfull = 2 * nquad - 1
        fiberfrac = np.zeros((nfull, nfull), np.float32)
        # Loop over offsets in the x >= 0 and y >= 0 quadrant.
        origin = nquad - 1
        reverse = slice(None, None, -1)
        for iy in range(nquad):
            for ix in range(nquad):
                T = self.offset_template[iy, ix]
                fiberfrac[origin + iy, origin + ix] = np.sum(P * T) / Psum
                if iy > 0:
                    # Calculate in the x >= 0 and y < 0 quadrant.
                    fiberfrac[origin - iy, origin + ix] = np.sum(P * T[reverse, :]) / Psum
                if ix > 0:
                    # Calculate in the x < 0 and y >= 0 quadrant.
                    fiberfrac[origin + iy, origin - ix] = np.sum(P * T[:, reverse]) / Psum
                if iy > 0 and ix > 0:
                    # Calculate in the x < 0 and y < 0 quadrant.
                    fiberfrac[origin - iy, origin - ix] = np.sum(P * T[reverse, reverse]) / Psum
        # Locate the best centered offset.
        iy, ix = np.unravel_index(np.argmax(fiberfrac), fiberfrac.shape)
        xc = self.xyoffset[ix]
        yc = self.xyoffset[iy]
        # Calculate the radius of each pixel in arcsecs relative to this center.
        radius = np.hypot((self.xgrid - xc) * self.pixel_size_um / self.plate_scales[0],
                          (self.ygrid - yc) * self.pixel_size_um / self.plate_scales[1]).reshape(-1)
        # Fill ivar-weighted histograms of flux versus angular radius.
        WZ, _ = np.histogram(radius, bins=self.angbins, weights=(P * W).reshape(-1))
        W, _ = np.histogram(radius, bins=self.angbins, weights=W.reshape(-1))
        # Calculate the circularized profile, normalized to 1 at (xc, yc).
        Z = np.divide(WZ, W, out=np.zeros_like(W), where=W > 0)
        fwhm = -1
        if Z[0] > 0:
            Z /= Z[0]
            # Find the first bin where Z <= 0.5.
            k = np.argmax(Z <= 0.5)
            if k > 0:
                # Use linear interpolation over this bin to estimate FWHM.
                s = (Z[k] - 0.5) / (Z[k] - Z[k - 1])
                fwhm = 2 * ((1 - s) * self.rang[k] + s * self.rang[k - 1])
        self.Z = Z
        self.xcbest = xc
        self.ycbest = yc
        return fwhm, np.max(fiberfrac)


class CenteredStamp(object):

    def __init__(self, stamp_size, inset, method='fiber'):
        self.inset = inset
        self.stamp_size = stamp_size
        self.inset_size = stamp_size - 2 * inset
        # Calculate the range of offsets to explore.
        self.dxy = np.arange(-inset, inset + 1)
        noffset = len(self.dxy)
        # Allocate memory for the templates to use.
        self.template = np.zeros((noffset, noffset, stamp_size, stamp_size))
        # Define the template profile.
        rfibersq = (0.5 * 107 / 15) ** 2
        if method == 'fiber':
            profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < rfibersq)
        elif method == 'donut':
            xymax = 0.5 * (stamp_size - 2 * inset)
            def profile(x, y):
                rsq = x ** 2 + y ** 2
                return rsq * np.exp(-0.5 * rsq / (3 * rfibersq)) * (np.maximum(np.abs(x), np.abs(y)) < xymax)
        else:
            raise ValueError('Unsupported method "{0}".'.format(method))
        # Build the profiles.
        for iy in range(noffset):
            for ix in range(noffset):
                self.template[iy, ix] = make_template(
                    stamp_size, profile, dx=self.dxy[ix], dy=self.dxy[iy], normalized=True)

    def center(self, D, W):
        assert D.shape == (self.stamp_size, self.stamp_size) and W.shape == D.shape
        S = slice(self.inset, self.inset + self.inset_size)
        # Calculate the weighted mean template flux for each offset.
        WDsum = np.sum(W * D * self.template, axis=(2, 3))
        Wsum = np.sum(W * self.template, axis=(2, 3))
        meanflux = np.divide(WDsum, Wsum, out=np.zeros(self.template.shape[:2]), where=Wsum > 0)
        # Calculate the best-centered offset
        iy, ix = np.unravel_index(np.argmax(meanflux.reshape(-1)), meanflux.shape)
        yslice = slice(iy, iy + self.inset_size)
        xslice = slice(ix, ix + self.inset_size)
        return yslice, xslice


def uncompress(filein, pathout=None, overwrite=True):
    """Uncompress a tile-compressed FITS file with an .fz extension.
    """
    filein = pathlib.Path(filein)
    if filein.suffix != '.fz':
        return
    if pathout is not None:
        pathout = filein.parent
    fileout = (pathout or filein.parent) / filein.stem
    if fileout.exists() and not overwrite:
        raise RuntimeError('Output exists and overwrite is False: {0}'.format(fileout))
    with fitsio.FITS(str(filein), mode='r') as IN:
        with fitsio.FITS(str(fileout), mode='rw', clobber=overwrite) as OUT:
            for hdu in IN:
                header = hdu.read_header()
                data = hdu.read()
                OUT.write(data, header=header, extname=hdu.get_extname())
    return str(fileout)


def ADCangles(EL, HA, DEC, LAT=31.963972222):
    """Calculate the parallactic angle in degrees W of N. Inputs in degrees."""
    Z, HA, coDEC, coLAT = np.deg2rad([90 - EL, HA, 90 - DEC, 90 - LAT])
    if Z == 0:
        return np.zeros(3)
    sinZ = np.sin(Z)
    sinP = np.sin(HA) * np.sin(coLAT) / sinZ
    cosP = (np.cos(coLAT) - np.cos(coDEC) * np.cos(Z)) / (np.sin(coDEC) * sinZ)
    P = np.arctan2(sinP, cosP)
    # Formulas from DESI-4957
    tanZ = np.tan(Z)
    HORIZON = P + 0.5 * np.pi
    ADC1 = HORIZON + (0.0353 + tanZ * (0.2620 + tanZ * 0.3563))
    ADC2 = HORIZON - (0.0404 + tanZ * (0.2565 + tanZ * 0.3576))
    return np.rad2deg([P, ADC1, ADC2])


def diskgrid(n, radius=1, alpha=2):
    """Distribute points over a disk with increasing density towards the center.

    Points are locally uniformly distributed according to the sunflower pattern
    https://demonstrations.wolfram.com/SunflowerSeedArrangements/

    A non-linear transformation of the radial coordinate controlled by alpha
    increases the density of points towards the center. Use alpha=0 for
    uniform density.

    Parameters
    ----------
    n : int
        Total number of points to use in the grid.
    radius : float
        Radius of the disk to fill.
    alpha : float
        Parameter controlling the increase of density towards the center of
        the disk, with alpha=0 corresponding to no increase.

    Returns
    -------
    tuple
        Tuple (x, y) of 2D points covering the disk.
    """
    # Golden ratio.
    phi = 0.5 * (np.sqrt(5) + 1)
    # Calculate coordinates of each point to uniformly fill the unit disk.
    k = np.arange(1, n + 1)
    theta = 2 * np.pi * k / phi ** 2
    r = np.sqrt((k - 0.5) / (n - 0.5))
    # Transform r to increase the density towards the center.
    if alpha > 0:
        r = (np.exp(alpha * r) - 1) / (np.exp(alpha) - 1)
    r *= radius
    return r * np.cos(theta), r * np.sin(theta)


def fit_spots(data, ivar, profile, area=1):
    """Fit images of a spot to estimate the spot flux and background level.

    All inputs are nominally 2D but can have other shapes as long as
    they broadcast correctly. Input arrays with >2 dimensions are assumed
    to have the pixels indexed along their last 2 dimensions.

    Parameters
    ----------
    data : array
        Array of shape (...,ny,nx) with the data to fit.
    ivar : array
        Array of shape (...,ny,nx) with the corresponding ivars.
    profile : array
        Array of shape (...,ny,nx) with the spot profile(s) to use.
    area : scalar or array
        Area of each pixel used to predict its background level as b * area.
        Either a scalar or an array of shape (...,ny, nx).

    Returns
    -------
    tuple
        Tuple (f, b, cov) where f and b are arrays of shape (...) and
        cov has shape (...,2,2) with elements [...,0,0] = var(f),
        [...,1,1] = var(b) and [...,0,1] = [...,1,0] = cov(f,b).
    """
    # Calculate the matrix elements for the linear problem
    # [ M11 M12 ] [ f ] = [ A1 ]
    # [ M12 M22 ] [ b ]   [ A2 ]
    M11 = np.sum(ivar * profile ** 2, axis=(-2, -1))
    M12 = np.sum(ivar * area * profile, axis=(-2, -1))
    M22 = np.sum(ivar * area ** 2, axis=(-2, -1))
    A1 = np.sum(ivar * data * profile, axis=(-2, -1))
    A2 = np.sum(ivar * data * area, axis=(-2, -1))
    # Solve the linear problem.
    det = M11 * M22 - M12 ** 2
    M11 /= det
    M12 /= det
    M22 /= det
    f = (M22 * A1 - M12 * A2)
    b = (M11 * A2 - M12 * A1)
    # Calculate the covariance of (f, b).
    cov = np.stack((np.stack((M22, -M12), axis=-1), np.stack((-M12, M11), axis=-1)), axis=-1)
    return f, b, cov
