"""Calibration methods.
"""
import numpy as np

import scipy.special

import fitsio

import desietcimg.fit


class CalibrationAnalysis(object):
    
    ZERO_MASK = 0
    DARK_MASK = 1
    DARK_VAR =  2

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
        self.fitok, self.avgbias, self.rdnoise, self.zerodata = self.fit_pedestal(
            raw, verbose=verbose)
        mask, self.pixbias = self.mask_defects(raw, self.avgbias, self.rdnoise, verbose=verbose)
        self.pixmask[mask] |= (1 << CalibrationAnalysis.ZERO_MASK)
        if refine == 'auto':
            refine = self.rdnoise / np.sqrt(len(raw)) < np.std(self.pixbias[self.pixmask == 0])
        if refine:
            self.rdnoise, self.zerodata = self.refine_noise(
                raw, self.rdnoise, self.pixmask, self.pixbias, verbose=verbose)
            self.avgbias = np.mean(self.pixbias)
        if verbose:
            print('{0} read noise = {1:.3f} ADU avg bias = {2:.3f} ADU'
                  .format(self.name, self.rdnoise, self.avgbias))
        self.have_zeros = True

    def process_darks(self, raw, refine='auto', verbose=True):
        if not self.have_zeros:
            raise RuntimeError('Must call process_zeros before process_darks.')
        if verbose:
            print('== {0} darks analysis:'.format(self.name))
        self.fitok, self.avgdark, self.stddark, self.darkdata = self.fit_pedestal(
            raw, verbose=verbose)
        mask, self.pixmu = self.mask_defects(raw, self.avgdark, self.stddark, nsig=50, verbose=verbose)
        self.pixmask[mask] |= (1 << CalibrationAnalysis.DARK_MASK)

        self.pixmu2, self.pixvar, self.pvalue = self.pixel_variance(raw, verbose=verbose)
        #self.pixmask[mask] |= (1 << CalibrationAnalysis.DARK_VAR)
        return

        if refine == 'auto':
            refine = self.stddark / np.sqrt(len(raw)) < np.std(self.pixmu[self.pixmask == 0])
        if refine:
            self.stddark, self.darkdata = self.refine_noise(
                raw, self.stddark, self.pixmask, self.pixmu, verbose=verbose)
            self.avgdark = np.mean(self.pixmu)

        if self.stddark <= self.rdnoise:
            if verbose:
                print('Unable to determine gain since std(dark) < std(zero).')
            self.gain = np.nan
            return False

        invgain = (self.stddark ** 2 - self.rdnoise ** 2) / (self.avgdark - self.avgbias)
        print('invgain', invgain)

    def save(self, name, overwrite=True):
        """
        """
        if not overwrite and os.path.exists(name):
            raise RuntimeError('File exists and overwrite is False: {0}.'.format(name))
        with fitsio.FITS(name, 'rw', clobber=overwrite) as hdus:
            # Write a primary HDU with only the metadata.
            meta = dict(
                NAME=self.name,
                NY=self.shape[0],
                NX=self.shape[1],
                AVGBIAS=self.avgbias,
                RDNOISE=self.rdnoise,
                AVGDARK=self.avgdark,
                STDDARK=self.stddark,
            )
            hdus.write(np.zeros((1,), dtype=np.float32), header=meta)
            # Write the pixel mask.
            hdus.write(self.pixmask, extname='MASK')
            # Write the pixel biases.
            hdus.write(self.pixbias, extname='BIAS')
            hdus.write(self.pixmu, extname='MU')
            hdus.write(self.pixmu2, extname='MU2')
            hdus.write(self.pixvar, extname='VAR')
            hdus.write(self.pvalue, extname='PVAL')
            # Write tables of pedestal data.
            hdus.write(self.zerodata, extname='ZERDAT')
            hdus.write(self.darkdata, extname='DRKDAT')

    @staticmethod
    def load(name):
        """Restore results previously written by :meth:`save`.
        """
        with fitsio.FITS(name, 'r') as hdus:
            meta = hdus[0].read_header()
            CA = CalibrationAnalysis(meta['NAME'], meta['NY'], meta['NX'])
            CA.avgbias = meta['AVGBIAS']
            CA.rdnoise = meta['RDNOISE']
            CA.avgdark = meta['AVGDARK']
            CA.stddark = meta['STDDARK']
            CA.pixmask[:] = hdus['MASK'].read()
            CA.pixbias = hdus['BIAS'].read().copy()
            CA.pixmu = hdus['MU'].read().copy()
            CA.pixmu2 = hdus['MU2'].read().copy()
            CA.pixvar = hdus['VAR'].read().copy()
            CA.pvalue = hdus['PVAL'].read().copy()
            CA.zerodata = hdus['ZERDAT'].read().copy()
            CA.darkdata = hdus['DRKDAT'].read().copy()
            return CA

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

        return result.success, mu, std, self.fitter.data

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

    def pixel_variance(self, raw, verbose=True):
        """
        """
        nexp, maxval = self.validate(raw)
        ny, nx = raw[0].shape
        pixmu = np.zeros((ny, nx), np.float32)
        pixvar = np.zeros((ny, nx), np.float32)
        pvalue = np.zeros((ny, nx), np.float32)
        for iy in range(ny):
            for ix in range(nx):
                X = raw[:, iy, ix].astype(np.float)
                pixmu[iy, ix] = np.median(X)
                pixvar[iy, ix] = np.var(X)
                if pixvar[iy, ix] > 0:
                    _, pvalue[iy, ix] = scipy.stats.shapiro(X)
        return pixmu, pixvar, pvalue

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
            hist, _ = np.histogram(D[mask == 0], bins=xedge)
            pixhist += hist

        xpix = 0.5 * (xedge[1:] + xedge[:-1])
        theta0 = np.array([pixhist.sum(), 0., std])

        result, ntot, mu, std = self.fitter.fit(-iwin, iwin + 1, pixhist, pixhist.sum(), 0., std)
        if verbose:
            print('refine: mu={0:.2f} ADU std={1:.2f} ADU frac={2:.2f}%'
                  .format(mu, std, 100 * ntot / raw.size))
            print('refine: {0}'.format(result.message))

        # Correct for the variance due to noisy pixel mean estimates.
        std *= np.sqrt(1 - 1 / nexp)

        return std, self.fitter.data
