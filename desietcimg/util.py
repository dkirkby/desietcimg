"""General purpose utilities for imaging analysis.
"""
import os

import numpy as np

import scipy.signal
import scipy.stats


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


def prepare(D, W=None, invgain=1.6, smoothing=5, verbose=False):
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
        Number of pixels for boxcar smoothing of D used to estimate
        the signal variance contribution.

    Returns
    -------
    tuple
        Tuple D, W of 2D numpy float32 arrays.
    """
    # Find any saturated pixels.
    maxval = np.iinfo(D.dtype).max
    print(D.dtype, D.shape, maxval)
    saturated = (D == maxval)
    if np.any(saturated):
        print(f'Found {np.count_nonzero(saturated)} saturated pixels.')
    # Convert to a float32 array.
    D = np.asarray(D, np.float32)
    # Select background pixels using sigma clipping.
    clipped, _, _ = scipy.stats.sigmaclip(D[~saturated])
    # Subtract the clipped mean from the data.
    bgmean = np.mean(clipped)
    if verbose:
        print('Subtracted background mean {0:.1f} ADU.'.format(bgmean))
    D -= bgmean
    if W is None:
        # Use the clipped pixels for the background variance estimate.
        bgvar = np.var(clipped)
        if verbose:
            print('Estimated background RMS {0:.1f} ADU.'.format(np.sqrt(bgvar)))
        var = bgvar * np.ones_like(D)
        # Estimate additional variance due to smoothed signal.
        smoother = np.ones((smoothing, smoothing)) / smoothing ** 2
        Dsmoothed = scipy.signal.convolve(D, smoother, mode='same')
        var += np.maximum(0., Dsmoothed) / invgain
        # Build an inverse variance image with zeros where var is zero.
        W = np.divide(1., var, out=np.zeros_like(var, dtype=np.float32), where=var > 0)
    else:
        W = np.asarray(W, dtype=np.float32)
    # Zero ivar for any saturated pixels.
    W[saturated] = 0.
    return D, W


def mask_defects(D, W, chisq_max=5e3, kernel_size=5, inplace=False):
    if not inplace:
        W = W.copy()
    # Initialize the kernel.
    if kernel_size % 2 == 0:
        raise ValueError('Kernel size must be odd.')
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    nby2 = kernel_size // 2
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
        if verbose:
            print(iy, ix, np.max(chisq))
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
    """Return the absolute path to a named data file.
    
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
