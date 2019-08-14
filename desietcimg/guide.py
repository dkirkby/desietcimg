import functools
import os.path

import numpy as np

import fitsio

import desietcimg.util
import desietcimg.fit


def moffat_profile(x, y, fwhm, sx=1, sy=1, beta=3.5):
    r0 = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    r = np.sqrt((x / sx) ** 2 + (y / sy) ** 2)
    return (1 + (r / r0) ** 2) ** -beta


class GuideCameraAnalysis(object):
    """Initialize the guide camera image analysis.

    Parameters
    ----------
    stamp_size : int
        Analysis will use square stamps with this pixel size. Must be odd.
    pixel_size_um : float
        The pixel  size in microns.
    plate_scales : tuple of two floats
        The nominal plate scales in microns / arcsec along the pixel x and y directions.
    match_fwhm_arcsec : float
        The nominal FWHM of a Moffat PSF with beta=3.5 used as a matched filter to detect
        PSF-like sources.
    """
    def __init__(self, stamp_size=65, pixel_size_um=9, plate_scales=(70., 76.), match_fwhm_arcsec=1.1):
        assert stamp_size % 2 == 1
        self.rsize = stamp_size // 2
        self.stamp_size = stamp_size
        # Build a nominal PSF model for detection matching.
        profile = functools.partial(moffat_profile, fwhm=match_fwhm_arcsec,
                                    sx=plate_scales[0] / pixel_size_um, sy=plate_scales[1] / pixel_size_um)
        self.PSF0 = desietcimg.util.make_template(stamp_size, profile, normalized=True)
        # Define pixel coordinate grids for moment calculations.
        dxy = np.arange(stamp_size) - 0.5 * (stamp_size - 1)
        self.xgrid, self.ygrid = np.meshgrid(dxy, dxy, sparse=False)
        # Initialize primary fitter.
        self.fitter = desietcimg.fit.GaussFitter(stamp_size)
        # Initialize a slower secondary fitter for when the primary fitter fails to converge.
        self.fitter2 = desietcimg.fit.GaussFitter(stamp_size, optimize_args=dict(
            method='Nelder-Mead', options=dict(maxiter=10000, xatol=1e-3, fatol=1e-3, disp=False)))
        self.stamps = None
        self.results = None

    def detect_sources(self, D, W=None, meta={}, nsrc=12,
        chisq_max=150., min_central=18, cdist_max=3., saturation=61000, verbose=False):
        """Detect PSF-like sources in an image.

        Parameters
        ----------
        D : array
            2D array of image pixel values with shape (ny, nx).
        W : array or None
            2D array of correponsding inverse-variance weights with shape (ny, nx).
            When None, this array will be estimated from D.
        meta : dict
            Metadata associated with this input image. Will be propagated to the results
            with NSRC and SSIZE appended.
        nsrc : int
            Number of candiate PSF sources to detect, in (roughly) decreasing order of SNR.
        chisq_max : float
            Cut on chisq value passed to :func:`desietcimg.util.mask_defects` for
            each candidate stamp.  Used to reject fakes due to hot pixels or cosmics.
        min_central : int
            Minimum number of unmasked pixels required in the central 5x5 region.
            Used to reject saturated stars.
        cdist_max : float
            Maximum distance of the image centroid from the stamp center.
            Used to reject the wings of bright stars.
        saturation : int
            Raw pixel values >= this level are considered saturated.

        Returns
        -------
        :class:`GuideCameraResults`
            Object containing the stamps, fit results and metadata for this detection.
        """
        D, W = desietcimg.util.prepare(D, W, saturation=saturation)
        if verbose:
            print('Input image has {0} masked pixels.'.format(np.count_nonzero(W == 0)))
        meta = dict(meta)
        meta['NSRC'] = nsrc
        meta['SSIZE'] = self.stamp_size
        # Mask the most obvious defects in the whole image with a very loose chisq cut.
        W, nmasked = desietcimg.util.mask_defects(D, W, chisq_max=1e4, min_neighbors=7, inplace=True)
        if verbose:
            print('Masked {0} defect pixels in full image.'.format(nmasked))
        ny, nx = D.shape
        h = self.rsize
        ss = self.stamp_size
        # Calculate the ivar-weighted image.
        WD = np.array(W * D, np.float32)
        # Convolve the image (WD and W) with the matched PSF filter.
        CWD = desietcimg.util.Convolutions([WD], self.PSF0)
        WD = CWD.sources[0]
        WDf = CWD.convolved[0]
        CW = desietcimg.util.Convolutions([W], self.PSF0)
        Wf = CW.convolved[0]
        filtered = np.divide(WDf, Wf, out=np.zeros_like(W), where=Wf > 0)
        inset = filtered[h:-h, h:-h]
        fmin = np.min(filtered)
        stamps, results = [], []
        while len(stamps) < nsrc:
            # Find the largest filtered value in the inset region and its indices [iy, ix].
            iy, ix = np.unravel_index(np.argmax(inset), (ny - 2 * h, nx - 2 * h))
            fmax = inset[iy, ix]
            
            # Extract the stamp centered on [iy, ix] in the full (non-inset) image.
            xlo, ylo = ix, iy
            xhi, yhi = ix + ss, iy + ss
            stamp = D[ylo:yhi, xlo:xhi].copy()
            ivar = W[ylo:yhi, xlo:xhi].copy()

            # Cosmic-ray detection.
            '''
            Dgrad = desietcimg.util.sobelfilter(stamp, ivar)
            n100 = np.count_nonzero(Dgrad > 100)
            n10 = np.count_nonzero(Dgrad > 10)
            '''
            
            # Find the largest filtered value outside of this stamp.
            save = filtered[ylo:yhi, xlo:xhi].copy()
            filtered[ylo:yhi, xlo:xhi] = fmin
            f2nd = np.max(inset)
            filtered[ylo:yhi, xlo:xhi] = save

            if verbose:
                print('Candidate at ({0},{1}) with fmax={2:.1f} f2nd={3:.1f}'
                      .format(xlo + h, ylo + h, fmax, f2nd))

            # Mask pixel defects in this stamp.
            ivar, nmasked = desietcimg.util.mask_defects(
                stamp, ivar, chisq_max=chisq_max, min_neighbors=5, inplace=True)
            if verbose:
                print('  Masked {0} pixels within stamp.'.format(nmasked))
            
            # Update the WD and W convolutions with the new ivar.
            CWD.set_source(slice(ylo, yhi), slice(xlo, xhi), stamp * ivar)
            changed = CW.set_source(slice(ylo, yhi), slice(xlo, xhi), ivar)
            # Calculate the updated filtered array.
            filtered[changed] = 0
            if verbose:
                print('  nchanged {0}'.format(np.count_nonzero(changed)))
                print('  WDf: min={0:.1f} max={1:.1f} nan? {2}'.format(
                    np.min(WDf[changed]), np.max(WDf[changed]), np.any(np.isnan(WDf[changed]))))
                print('  Wf: min={0:.1f} max={1:.1f} nan? {2}'.format(
                    np.min(Wf[changed]), np.max(Wf[changed]), np.any(np.isnan(Wf[changed]))))
            filtered[changed] = np.divide(
                WDf[changed], Wf[changed], out=filtered[changed], where=Wf[changed] > 0)

            # Calculate the change in the filtered value after the masking.
            # We can assume that the denominator is non-zero here.
            wsum = np.sum(ivar * self.PSF0)
            fnew = 0
            if wsum > 0:
                fnew = np.sum(stamp * ivar * self.PSF0) / wsum
            if fnew < f2nd:
                # This stamp had a artificially high filtered value due to pixel
                # defects so skip it now.  It might still come back at a lower
                # value after masking.
                if verbose:
                    print('  Skipped with fnew={0:.1f} < f2nd={1:.1f}'.format(fnew, f2nd))
                # This shouldn't be necessary but prevents 13047-CIW from getting into a loop
                # where filtered[iy, ix] is not changed by anything above.
                inset[iy, ix] = 0
                continue

            # Redo the filtering with the data in this stamp set to zero
            # so we don't pick this stamp again.
            changed = CWD.set_source(slice(ylo, yhi), slice(xlo, xhi), 0)
            filtered[changed] = 0
            filtered[changed] = np.divide(
                WDf[changed], Wf[changed], out=filtered[changed], where=Wf[changed] > 0)

            # Count the number of unmasked pixels in the central 5x5.
            nwindow = 2
            c_slice = slice(h - nwindow, h + nwindow + 1)
            ncentral = np.sum(ivar[c_slice, c_slice] > 0)
            if ncentral < min_central:
                # This stamp has too many masked pixels in the central core to
                # useful. This is necessary to reject saturated stars.
                if verbose:
                    print('  Dropped with ncentral={0} < {1}.'.format(ncentral, min_central))
                continue

            # Stamps are sometimes selected on the wings of a previously selected
            # bright star.  To detect this condition, calculate the stamp's
            # centroid (w/o centered PSF weights).
            clipped = np.maximum(0., stamp)
            M0 = np.sum(clipped * ivar)
            Mx = np.sum(self.xgrid * clipped * ivar) / M0
            My = np.sum(self.ygrid * clipped * ivar) / M0
            
            # Calculate the centroid distance from the stamp center.
            cdist = np.sqrt(Mx ** 2 + My ** 2)

            # Ignore stamps whose centroid is not centered, which probably indicates
            # we have found the wing of a previously found bright star.
            if cdist > cdist_max:
                if verbose:
                    print('  Dropped with cdist={0:.2f} > {1:.2f}.'.format(cdist, cdist_max))
                continue

            # Fit a single Gaussian + constant background to this stamp.
            result = self.fitter.fit(stamp, ivar)
            if verbose:
                print('  Fit: {0}'.format(result['message']))
            if not result['success']:
                result = self.fitter2.fit(stamp, ivar)
                if verbose:
                    print('  2nd Fit: {0}'.format(result['message']))
    
            # Save this candidate PSF-like source.
            stamps.append((stamp, ivar))
            results.append((result, slice(ylo, yhi), slice(xlo, xhi)))

        results = self.select_psf(results, verbose=verbose)
        profile = self.get_psf_profile(stamps, results)

        return GuideCameraResults(stamps, results, profile, meta)

    def  select_psf(self, results, smin=2.0, gmax=0.25, rmax=1.0, sscale=0.5, gscale=0.05, nbright=4, verbose=False):
        """Select the PSF-like  sources.

        Results are stored as a boolean in the 'psf' attribute.
        """
        nsrc = len(results)
        svec = np.zeros(nsrc)
        gvec = np.ones(nsrc)
        rvec = np.zeros(nsrc)
        snrvec = np.zeros(nsrc)
        for k, (fit, yslice, xslice) in enumerate(results):
            if not fit['success']:
                continue
            svec[k] = fit['s']
            gvec[k] = np.hypot(fit['g1'], fit['g2'])
            rvec[k] = np.hypot(fit['x0'], fit['y0'])
            snrvec[k] = fit['snr']
        # Identify the PSF candidates.
        cand = (svec > smin) & (gvec < gmax) & (rvec < rmax)
        # Pick the (up to) nbright brightest PSF candidates.
        snrvec[~cand] = 0
        brightest = np.argsort(snrvec)[-nbright:]
        # Get the median size and ellipticity of the brightest candidates.
        smed = np.median(svec[brightest])
        gmed = np.median(gvec[brightest])
        if verbose:
            print('  Selected {0} PSF candidates with median s = {1:.1f}, g = {2:.3f} of {3} brightest'
                  .format(np.count_nonzero(cand), smed, gmed, len(brightest)))
        # Select all candidates that are close enough to the median.
        dist = ((svec - smed) / sscale) ** 2 + ((gvec - gmed) / gscale) ** 2
        psf = cand & (dist < 1)
        if verbose:
            print('  Selected {0} final PSF candidates.'.format(np.count_nonzero(psf)))
        for k, (fit, yslice, xslice) in enumerate(results):
            fit['psf'] = bool(psf[k])
        return results

    def get_psf_profile(self, stamps, results):
        """Stack PSF-like detected sources to estimate the normalized PSF profile.
        """
        WPsum = np.zeros((self.stamp_size, self.stamp_size))
        Wsum = np.zeros((self.stamp_size, self.stamp_size))
        for k, (D, W) in enumerate(stamps):
            fit, yslice, xslice = results[k]
            if not fit['psf']:
                continue
            b, f = fit['b'], fit['f']
            WPsum += W * (D - b) * f
            Wsum += f ** 2 * W
        P = np.divide(WPsum, Wsum, out=np.zeros_like(Wsum), where=Wsum > 0)
        return P, Wsum


class GuideCameraResults(object):
    """Container for guide camera analysis results.
    """
    def __init__(self, stamps, results, profile, meta):
        self.stamps = stamps
        self.results = results
        self.profile = profile
        self.meta = meta

    def print(self):
        rsize = self.meta['SSIZE'] // 2
        for k in range(self.meta['NSRC']):
            result, y_slice, x_slice = self.results[k]
            print('SRC{0} [x={1}, y={2}]'.format(
                k, x_slice.start + rsize, y_slice.start + rsize))
            if result['success']:
                print('  SNR={0:.1f}'.format(result['snr']), end='')
                for pname in desietcimg.fit.GaussFitter.pnames:
                    print(', {0}={1:.3f}'.format(pname, result[pname]), end='')
                print()

    def save(self, name, overwrite=True):
        """Save results to a FITS file.

        Parameters
        ----------
        name : str
            Name of the FITS file to write.
        overwrite : bool
            OK to silently overwrite any existing file when True.
        """
        if not overwrite and os.path.exists(name):
            raise RuntimeError('File exists and overwrite is False: {0}.'.format(name))
        with fitsio.FITS(name, 'rw', clobber=overwrite) as hdus:
            # Write a primary HDU with only the metadata.
            hdus.write(np.zeros((1,), dtype=np.float32), header=self.meta)
            # Write each stamp.
            for k in range(self.meta['NSRC']):
                result, y_slice, x_slice = self.results[k]
                hdr = dict(X1=x_slice.start, X2=x_slice.stop, Y1=y_slice.start, Y2=y_slice.stop)
                for key, val in result.items():
                    hdr[key.upper() + '_'] = val
                hdus.write(np.stack(self.stamps[k]), header=hdr, extname='SRC{0}'.format(k))
            # Write the profile.
            hdus.write(np.stack(self.profile), extname='PROFILE')

    @staticmethod
    def load(name):
        """Restore results previously written by :meth:`save`.
        """
        with fitsio.FITS(name, 'r') as hdus:
            stamps, results = [], []
            meta = dict(hdus[0].read_header())
            for k in range(meta['NSRC']):
                extname = 'SRC{0}'.format(k)
                hdr = dict(hdus[extname].read_header())
                x_slice = slice(hdr['X1'], hdr['X2'])
                y_slice = slice(hdr['Y1'], hdr['Y2'])
                result = {}
                for key, val in hdr.items():
                    if key.endswith('_'):
                        result[key[:-1].lower()] = val
                results.append((result, y_slice, x_slice))
                data = hdus[extname].read()
                stamps.append((data[0].copy(), data[1].copy()))
            data = hdus['PROFILE'].read()
            profile = (data[0].copy(), data[1].copy())
        return GuideCameraResults(stamps, results, profile, meta)
