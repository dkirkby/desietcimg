"""Calibration methods.
"""
import numpy as np

import scipy.special
import scipy.linalg

import fitsio

import desietcimg.fit


class CalibrationAnalysis(object):
    
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
        self.have_darks = False
        self.have_flats = False

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
        ok = self.check_consistency(raw, verbose=verbose)
        nok = np.count_nonzero(ok)
        fitok, self.avgbias, self.rdnoise, self.zerodata = self.fit_pedestal(
            raw[ok], verbose=verbose)
        mask, self.pixbias = self.mask_defects(raw[ok], self.avgbias, self.rdnoise, verbose=verbose)
        self.pixmask[mask] |= (1 << CalibrationAnalysis.ZERO_MASK)
        if refine == 'auto':
            refine = self.rdnoise / np.sqrt(nok) < np.std(self.pixbias[self.pixmask == 0])
        if refine:
            self.rdnoise, self.zerodata = self.refine_noise(
                raw[ok], self.rdnoise, self.pixmask, self.pixbias, verbose=verbose)
            self.avgbias = np.mean(self.pixbias)
        if verbose:
            print('{0} read noise = {1:.3f} ADU avg bias = {2:.3f} ADU'
                  .format(self.name, self.rdnoise, self.avgbias))
        self.have_zeros = True

    def process_darks(self, raw, temperature, exptime, verbose=True):
        if not self.have_zeros:
            raise RuntimeError('Must call process_zeros before process_darks.')
        if verbose:
            print('== {0} darks analysis:'.format(self.name))
        ok = self.check_consistency(raw, self.pixmask > 0, verbose=verbose)
        fitok, self.avgdark, self.stddark, _ = self.fit_pedestal(raw[ok], verbose=verbose)
        mask, self.pixmu = self.mask_defects(raw, self.avgdark, self.stddark, nsig=50, verbose=verbose)
        self.pixmask[mask] |= (1 << CalibrationAnalysis.DARK_MASK)
        self.dark_temperature = temperature
        self.darkdata = self.dark_current_analysis(raw, exptime)
        self.have_darks = True

    def process_flats(self, raw, gain_guess=1.5, downsampling=32, verbose=True):
        """
        """
        if not self.have_zeros:
            raise RuntimeError('Must call process_zeros before process_darks.')
        if verbose:
            print('== {0} flats analysis:'.format(self.name))
        self.validate(raw)
        nexp, ny, nx = raw.shape
        FFF = desietcimg.fit.FlatFieldFitter((ny, nx), downsampling=downsampling)
        mu_all, var_all = [], []
        for iexp in range(nexp):
            result, ypred = FFF.fit(raw[iexp], self.pixmask, self.pixbias,
                                    self.rdnoise, gain_guess, verbose=verbose)
            mu, var = self.flat_gain_analysis(
                raw[iexp] - self.pixbias, self.pixmask, ypred, self.rdnoise)
            mu_all.append(mu)
            var_all.append(var)
        # Perform a least squares fit to the slope of these poihts, constrained to pass
        # through the origin, to estimate the inverse gain.
        mu = np.concatenate(mu_all)
        var = np.concatenate(var_all)
        self.flatinvgain = 1 / np.linalg.lstsq(mu.reshape(-1, 1), var, rcond=None)[0][0]
        # Package the mu,var values into a recarray.
        self.flatdata = np.empty(len(mu), dtype=[('mu', np.float32), ('var', np.float32),])
        self.flatdata['mu'] = mu
        self.flatdata['var'] = var
        self.have_flats = True

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
                DARKS=self.have_darks,
                FLATS=self.have_flats,
            )
            if self.have_darks:
                meta.update(dict(
                    AVGDARK=self.avgdark,
                    STDDARK=self.stddark,
                    DKTEMP=self.dark_temperature,
                ))
            if self.have_flats:
                meta.update(dict(
                    FLATG=self.flatinvgain,
                ))
            hdus.write(np.zeros((1,), dtype=np.float32), header=meta)
            # Write the pixel mask.
            hdus.write(self.pixmask, extname='MASK')
            # Write image data.
            hdus.write(self.pixbias, extname='BIAS')
            if self.have_darks:
                hdus.write(self.pixmu, extname='MU')
            # Write table data.
            hdus.write(self.zerodata, extname='ZERDAT')
            if self.have_darks:
                hdus.write(self.darkdata, extname='DRKDAT')
            if self.have_flats:
                hdus.write(self.flatdata, extname='FLTDAT')

    @staticmethod
    def load(name):
        """Restore results previously written by :meth:`save`.
        """
        with fitsio.FITS(name, 'r') as hdus:
            meta = hdus[0].read_header()
            CA = CalibrationAnalysis(meta['NAME'], meta['NY'], meta['NX'])
            CA.have_zeros = True
            CA.have_darks = meta['DARKS']
            CA.have_flats = meta['FLATS']
            CA.avgbias = meta['AVGBIAS']
            CA.rdnoise = meta['RDNOISE']
            CA.pixmask[:] = hdus['MASK'].read()
            CA.pixbias = hdus['BIAS'].read().copy()
            CA.zerodata = hdus['ZERDAT'].read().copy()
            if CA.have_darks:
                CA.avgdark = meta['AVGDARK']
                CA.stddark = meta['STDDARK']
                CA.dark_temperature = meta['DKTEMP']
                CA.pixmu = hdus['MU'].read().copy()
                CA.darkdata = hdus['DRKDAT'].read().copy()
            if CA.have_flats:
                CA.flatinvgain = meta['FLATG']
                CA.flatdata = hdus['FLTDAT'].read().copy()
            return CA

    def check_consistency(self, raw, mask=None, threshold=0.25, verbose=True):
        """
        """
        nexp, maxval = self.validate(raw)
        if nexp < 3:
            if verbose:
                print('Cannot check consistency with nexp < 3.')
            return np.ones(nexp, bool)
        # Calculate percentiles for each exposure.
        pctile = np.empty((nexp, 3))
        for iexp in range(nexp):
            D = raw[iexp].copy()
            if mask is not None:
                D = D[~mask]
            pctile[iexp] = np.percentile(D,  q=(25, 50, 75))
        lo, med, hi = pctile.T
        # Look for outliers in the median, using hi - lo to set the scale.
        dev = np.abs(med -  np.median(med)) / np.median(hi - lo)
        ok =  dev < threshold
        if verbose:
            nok = np.count_nonzero(ok)
            print('{0} / {1} exposures fail consistency check with threshold={2}.'
                  .format(nexp - nok, nexp, threshold))
            print('Max consistency deviation is {0:.4f} < threshold={1}.'
                  .format(np.max(dev), threshold))
        return ok

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

    def flat_gain_analysis(self, ydata, mask, ypred, rdnoise, nsig_cut=0.05, nbin=10, verbose=True):
        """Estimate variance in bins of bias-subtracted signal.

        Parameters
        ----------
        ydata : array
            2D array of bias-subtracted data from a single exposure in ADU.
        mask : array
            2D array of pixel masks where zero indicates a good pixel to
            use in the analysis.
        ypred : array
            2D array of smooth predicted bias-subtracted data in ADU.
            Can be obtained using :class:`desietcimg.fit.FlatFieldFitter`.

        Returns
        -------
        tuple
            Tuple (mu, var) of mean predicted bias-subtracted signal in ADU
            and the corresponding shot-noise variance in ADU ** 2 estimated in
            bins of predicted signal.
        """
        valid = (mask == 0)
        ypred_valid = ypred[valid]
        residual = ydata - ypred

        # Calculate percentile bins in the predicted value.
        edges = np.percentile(ypred_valid, np.linspace(0, 100, nbin + 1))
        binsize = np.diff(edges)
        ibin = np.full(ydata.shape, -1)
        ibin[valid] = np.minimum(nbin - 1, np.digitize(ypred_valid, edges, right=False) - 1)    

        # Calculate the mean and variance in each bin.
        mu, var = [], []
        for i in range(nbin):
            if binsize[i] > 2 * np.median(binsize):
                # Do not use bins that are too wide.
                continue
            inbin = ibin == i
            Y = ypred[valid & inbin]
            DY = residual[valid & inbin]
            _, lo, hi = scipy.stats.sigmaclip(DY, low=6, high=6)
            good = (DY >= lo) & (DY <= hi)
            mu.append(np.mean(Y[good]))
            var.append(np.var(DY[good]) - rdnoise ** 2)
        return np.array(mu), np.array(var)

    def dark_current_analysis(self, raw, exptime, nbins=200, clip=(0.1, 90), verbose=True):

        nexp, maxval = self.validate(raw)

        # Set analysis range from pixel values.
        valid = self.pixmask == 0
        lo, hi = maxval, 0
        for iexp in range(nexp):
            lo_exp, hi_exp = np.percentile(raw[iexp, valid], clip)
            lo = min(lo, lo_exp)
            hi = max(hi, hi_exp)
        lo = lo - self.avgbias
        hi = hi - self.avgbias
        bins = np.linspace(lo, hi, nbins + 1)

        # Histogram the means.
        signal = self.pixmu - self.pixbias
        mean_hist, _ = np.histogram(signal, bins)

        # Histogram each exposure.
        exp_hist = np.empty((nexp, nbins), int)
        for iexp in range(nexp):
            exp_hist[iexp], _ = np.histogram((raw[iexp] - self.pixbias)[valid], bins)
        # Calculate the per-pixel median of the exposure histograms.
        one_hist = np.median(exp_hist, axis=0)

        # Find the best fit model.
        xbin = 0.5 * (bins[1:] + bins[:-1])
        ##result, yfit = fit(xbin, mean_hist)

        data = np.empty(len(xbin), dtype=[('xbin', np.float32),
                                        ('yexp', np.float32),
                                        ('yavg', np.float32),
                                        ('yfit', np.float32)])
        data['xbin'] = xbin
        data['yexp'] = one_hist
        data['yavg'] = mean_hist
        #data['yfit'] = yfit

        return data
        #return result, data