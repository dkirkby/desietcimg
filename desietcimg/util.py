"""General purpose utilities for imaging analysis.
"""
import os
import re
import pathlib
import logging

import numpy as np

import scipy.signal
import scipy.stats

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


def preprocess(D, W, nsig_lo=8, nsig_hi=24, vmin=None, vmax=None):
    """Preprocess weighted 2D array data.
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
        raise ValueError('Missing sequence number {{N}} in pattern: {0}'
                         .format(file_pattern))
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
            if min is not None and N< min:
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
    if '{N}' in files:
        files = find_files(files)
    nexp = len(files)
    if slices is None:
        slices = (slice(None), slice(None))
    meta = {key: [] for key in keys}
    for k, file in enumerate(files):
        with fitsio.FITS(file, mode='r') as hdus:
            if k == 0:
                data = hdus[hdu][slices]
                raw = np.empty((nexp,) + data.shape, data.dtype)
                raw[0] = data
                logging.info('Reading {0} files with shape {1} and dtype {2}.'
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
