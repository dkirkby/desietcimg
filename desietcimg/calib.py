"""Calibration methods.
"""
import numpy as np

import scipy.special

import desietcimg.fit


class Calibrator(object):
    
    ZERO_MASK = 0
    DARK_MASK = 1

    def __init__(self, name, ny, nx):
        self.name = name
        self.shape = (ny, nx)
        self.pixmask = np.zeros(self.shape, np.uint8)
        self.fitter = desietcimg.fit.CalibFitter()
        self.reset()
        
    def reset(self):
        self.pixmask[:] = 0
        self.have_zeros = False
        
    def validate(self, raw):
        if raw.dtype != np.uint16:
            raise ValueError('Raw data must be uint16.')
        maxval = np.iinfo(raw.dtype).max
        nexp, ny, nx = raw.shape
        if (ny, nx) != self.shape:
            raise ValueError('Raw data has shape {0} but expected {1}.'
                             .format((ny, nx), self.shape))
        return nexp, maxval
        
    def process_zeros(self, raw, refine='auto', verbose=True):
        """Process a sequence of zero-length exposures to estimate bias and readnoise.
        
        Parameters
        ----------
        raw : array
            Array of shape (nexp, ny, nx) with uint16 datatype.
        refine : bool or 'auto'
            When True, refine the estimated read noise using per-pixel bias estimates
            and dropping masked pixels. This takes significantly longer than the rest
            of the analysis and is not worth running unless the per-pixel bias estimates
            are sufficiently accurate relative to their true dispersion. Use 'auto' to
            only refine the read noise when std(pixbias) > std(pix) / np.sqrt(nexp),
            where std(pix) is the initial read noise estimate that assumes a constant
            bias level.
        verbose : bool
            Print verbose progress and results.
        """
        if verbose:
            print('== {0} zeros analysis:'.format(self.name))
        self.fitok, self.avgbias, self.rdnoise = self.fit_pedestal(raw, verbose=verbose)
        mask, self.pixbias = self.mask_defects(raw, self.avgbias, self.rdnoise, verbose=verbose)
        self.pixmask |= (1 << Calibrator.ZERO_MASK)
        if refine == 'auto':
            refine = self.rdnoise / np.sqrt(len(raw)) < np.std(self.pixbias)
        if refine:
            self.rdnoise = self.refine_noise(raw, self.rdnoise, self.mask, self.pixbias, verbose=verbose)
            self.avgbias = np.mean(self.pixbias)
        if verbose:
            print('{0} read noise = {1:.3f} ADU avg bias = {2:.3f} ADU'
                  .format(self.name, self.rdnoise, self.avgbias))
        self.have_zeros = True

    def process_darks(self, raw, verbose=True):
        if not self.have_zeros:
            raise RuntimeError('Must call process_zeros before process_darks.')
        if verbose:
            print('== {0} darks analysis:'.format(self.name))
        self.fitok, self.avgdark, self.stddark = self.fit_pedestal(raw, verbose=verbose)
        mask, self.pixmu = self.mask_defects(raw, self.avgdark, self.stddark, verbose=verbose)
        self.pixmask |= (1 << Calibrator.DARK_MASK)
        
        if self.stddark <= self.rdnoise:
            if verbose:
                print('Unable to determine gain since std(dark) < std(zero).')
            self.gain = np.nan
            return False

        gain = (self.avgdark - self.avgbias) / (self.stddark ** 2 - self.rdnoise ** 2)
        print('gain', gain)

    def fit_pedestal(self, raw, nsiglo=3, nsighi=1, verbose=True):
        """
        """
        nexp, maxval = self.validate(raw)

        # Histogram the raw pixel values.
        pixhist = np.bincount(raw.reshape(-1), minlength=maxval)
        ntot = np.sum(pixhist)
        
        # Find the mode of the pixel value distribution.
        mode = np.argmax(pixhist)

        # Calculate the variance of the bins below the mode.
        wgt = pixhist[:mode]
        wsum = np.sum(wgt)
        dx = -np.arange(mode) + mode
        var = np.sum(wgt * dx ** 2) / wsum
        std = np.sqrt(var)

        # Fit the region from mode - nsiglo * std to mode + nsighi * std to a single Gaussian.
        ilo = int(np.floor(mode - nsiglo * std))
        ihi = int(np.ceil(mode + nsighi * std)) + 1        
        result, ntot, mu, std = self.fitter.fit(ilo, ihi, pixhist[ilo:ihi], ntot, mode, std)
        if verbose:
            print('fit: mu={0:.2f} ADU std={1:.2f} ADU frac={2:.2f}%'
                  .format(mu, std, 100 * ntot / raw.size))
            print('fit: {0}'.format(result.message))
        return result.success, mu, std
        
    def mask_defects(self, raw, mu, std, nsig=5, verbose=True):
        """
        """
        nexp, maxval = self.validate(raw)
        
        # Classify pixels more than nsig std from the mean as defects (dead / hot / cosmic).
        imin = np.uint16(max(0, int(np.floor(mu - nsig * std))))
        imax = np.uint16(min(maxval - 1, int(np.ceil(mu + nsig * std))))
        
        # Loop over frames to count how often a pixel is above or below range.
        D = np.empty_like(raw[0])
        nlo = np.zeros(self.shape, np.uint16)
        nhi = np.zeros(self.shape, np.uint16)
        psum = np.zeros(self.shape, np.uint32)
        for iexp in range(nexp):
            D[:] = raw[iexp]
            lo = D <= imin
            nlo[lo] += 1
            hi = D >= imax
            nhi[hi] += 1
            # Zero out any hi values assuming these are either defects or cosmics.
            D[hi] = 0
            psum += D

        # Classify pixels that are ever lo or hi in more than 1 frame (to allow for cosmics) as defects.
        mask = (nlo > 0) | (nhi > 1)
        
        if verbose:
            npix = np.prod(self.shape)
            print('defects: lo {0:.3f}% hi {1:.3f}%'
                  .format(100 * (nlo > 0).sum() / npix, 100 * (nhi > 1).sum() / npix))

        # Calculate the mean of each pixel value (ignoring hi values as likely cosmics)
        nsum = nexp - nhi
        pixmu = np.divide(psum, nsum, out=np.zeros(self.shape, np.float32), where=nsum > 0)
        if verbose:
            print('std(pixmu)={0:.3f} ADU std(pix)/sqrt(nexp) = {1:.3f} ADU'
                  .format(np.std(pixmu[~mask]), std / np.sqrt(nexp)))
        
        return mask, pixmu
    
    def refine_noise(self, raw, std, mask, pixmu, verbose=True):
        """
        """
        nexp, maxval = self.validate(raw)

        # Bin deviations from pixel means for a better estimate of the noise.
        iwin = int(np.ceil(3 * std))
        xedge = np.arange(-iwin, iwin + 2) - 0.5
        pixhist = np.zeros(2 * iwin + 1)
        D = np.empty(self.shape, np.float32)
        for iexp in range(nexp):
            D[:] = raw[iexp]
            D -= pixmu
            hist, _ = np.histogram(D[~mask], bins=xedge)
            pixhist += hist
            
        xpix = 0.5 * (xedge[1:] + xedge[:-1])
        plt.plot(xpix, pixhist, 'k.')
        theta0 = np.array([pixhist.sum(), 0., std])
        
        result, ntot, mu, std = self.fitter.fit(-iwin, iwin + 1, pixhist, pixhist.sum(), 0., std)
        if verbose:
            print('refine: mu={0:.2f} ADU std={1:.2f} ADU frac={2:.2f}%'
                  .format(mu, std, 100 * ntot / raw.size))
            print('refine: {0}'.format(result.message))

        plt.plot(xpix, self.fitter.yfit, 'r-', alpha=0.5)

        # Correct for the variance due to noisy pixel mean estimates.
        std *= np.sqrt(1 - 1 / nexp)
            
        return std
