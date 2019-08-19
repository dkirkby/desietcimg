import functools
import os.path

import numpy as np
import scipy.signal

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
    fiber_diam_um : float
        The fiber diameter in microns.
    pixel_size_um : float
        The pixel size in microns.
    plate_scales : tuple of two floats
        The nominal plate scales in microns / arcsec along the pixel x and y directions.
    match_fwhm_arcsec : float
        The nominal FWHM of a Moffat PSF with beta=3.5 used as a matched filter to detect
        PSF-like sources.
    nangbins : int
        Number of angular bins to use for FWHM and fiber fraction calculations.
    max_offset : float
        Maximum centroid offset in arcseconds to consider.
    noffset_per_pix : int
        Number of centroid offsets per pixel width to consider.
    """
    def __init__(self, stamp_size=65, fiber_diam_um=107., pixel_size_um=9, plate_scales=(70., 76.),
                 match_fwhm_arcsec=1.1, nangbins=40, max_offset=2.0, noffset_per_pix=2):
        assert stamp_size % 2 == 1
        self.rsize = stamp_size // 2
        self.stamp_size = stamp_size
        self.plate_scales = plate_scales
        self.fiber_diam_um = fiber_diam_um
        self.pixel_size_um = pixel_size_um
        self.noffset_per_pix = noffset_per_pix
        # Build a nominal PSF model for detection matching.
        profile = functools.partial(moffat_profile, fwhm=match_fwhm_arcsec,
                                    sx=plate_scales[0] / pixel_size_um, sy=plate_scales[1] / pixel_size_um)
        self.PSF0 = desietcimg.util.make_template(stamp_size, profile, normalized=True)
        # Define pixel coordinate grids for moment calculations.
        dxy = np.arange(stamp_size) - 0.5 * (stamp_size - 1)
        self.xgrid, self.ygrid = np.meshgrid(dxy, dxy, sparse=False)
        # Calculate coordinates of pixel centers in arcsecs.
        self.xang_pix = self.xgrid * self.pixel_size_um / self.plate_scales[0]
        self.yang_pix = self.ygrid * self.pixel_size_um / self.plate_scales[1]
        self.rang_pix = np.hypot(self.xang_pix, self.yang_pix).reshape(-1)
        # Specify angular binning for profile and FWHM calculations.
        rmax = dxy[-1] * self.pixel_size_um / max(self.plate_scales)
        self.angbins = np.linspace(0., rmax, nangbins + 1)
        self.profile_tab = np.zeros(nangbins, dtype=[('rang', np.float32), ('prof', np.float32)])
        self.profile_tab['rang'] = 0.5 * (self.angbins[1:] + self.angbins[:-1])
        # Calculate the maximum centroid offset that keeps the fiber within the stamp.
        max_offset_contained = int(np.floor(0.5 * (stamp_size - fiber_diam_um / pixel_size_um)))
        # Calculate the requested maximum centroid offset in pixels.
        max_offset_pix = int(np.ceil(max_offset * max(plate_scales) / pixel_size_um))
        if max_offset_pix > max_offset_contained:
            raise ValueError('max_offset = {0}" is not fully contained within stamp_size = {1}.'
                             .format(max_offset, stamp_size))
        # Build a grid of offsets in pixels for the x >= 0 and y >=0 quadrant.
        noffset = max_offset_pix * noffset_per_pix + 1
        xyoffset = np.linspace(0, +max_offset_pix, noffset)
        # Tabulate fiber templates for each (x,y) offset in the x >= 0 and y >= 0 quadrant.
        self.offset_template = np.empty((noffset, noffset, stamp_size, stamp_size), np.float32)
        max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
        profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < max_rsq)
        for iy in range(noffset):
            for ix in range(noffset):
                self.offset_template[iy, ix] = desietcimg.util.make_template(
                    stamp_size, profile, dx=xyoffset[ix], dy=xyoffset[iy], normalized=False)
        # Save a grid of offsets in pixels covering all quadrants.
        self.xyoffset = np.linspace(-max_offset_pix, +max_offset_pix, 2 * noffset - 1)
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
        meta['FIBSIZ'] = self.fiber_diam_um
        meta['PIXSIZ'] = self.pixel_size_um
        meta['XSCALE'] = self.plate_scales[0]
        meta['YSCALE'] = self.plate_scales[1]
        meta['NOFFPX'] = noffset_per_pix
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

        results, psf = self.select_psf(results, verbose=verbose)
        if np.any(psf):
            # Calculate the stacked PSF profile.
            profile = self.get_psf_profile(stamps, results)
            # Calcualte the fiber acceptance on a grid of (x,y) centroid offsets.
            fiberfrac = self.calculate_fiberfrac(profile)
            meta['FFRAC'] = np.max(fiberfrac)
            fwhm, circularized, xc, yc = self.calculate_fwhm(profile, fiberfrac)
            meta['FWHM'] = fwhm
            meta['XC'] = xc
            meta['YC'] = yc
            self.profile_tab['prof'] = circularized
        else:
            meta['FFRAC'] = -1.
            meta['FWHM'] = -1.
            meta['XC'] = 0.
            meta['YC'] = 0.
            profile = np.zeros_like(self.stamps[0][0]), np.zeros_like(self.stamps[0][0])
            self.profile_tab['prof'] = 0.
        if verbose:
            print('  NPSF = {0} FWHM = {1:.2f}" FIBERFRAC = {2:.3f}'.format(
                np.count_nonzero(psf), meta['FWHM'], meta['FFRAC']))

        return GuideCameraResults(stamps, results, profile, fiberfrac, self.profile_tab, meta)

    def  select_psf(self, results, smin=2.0, gmax=0.25, rmax=2.0, snrmin=30.,
                    dsmax=1.5, dgmax=0.10, nbright=5, verbose=False):
        """Select the PSF-like sources.

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
        # Identify the PSF candidates with loose cuts to reject cosmics and faint sources.
        cand = (svec > smin) & (gvec < gmax) & (rvec < rmax) & (snrvec > snrmin)
        if verbose:
            print('Rejected stamps: {0}'.format(np.where(~cand)[0]))
        psf = np.zeros_like(cand)
        if np.any(cand):
            # Select the PSF candidates with SNR within a factor of 5 of the brightest candidate.
            snrmax = np.max(snrvec[cand])
            brightest = (snrvec >= snrmax / 5) & cand
            # Use at most nbright bright candidates.
            brightest = np.where(brightest)[0][:nbright]
            if verbose:
                print('Brightest candidates: {0}'.format(brightest))
            # Get the minimum size and median ellipticity of these brightest candidates.
            smin = np.min(svec[brightest])
            gmed = np.median(gvec[brightest])
            if verbose:
                print('Brightest have min s = {0:.2f}, median g = {1:.3f}'.format(smin, gmed))
            # Select all candidates that are close enough these values.
            psf = cand & (svec < smin + dsmax) & (gvec < gmed + dgmax)
            if not np.any(psf):
                # Assume that the brightest candidate is a PSF.
                if verbose:
                    print('Fallback to using brightest candidate.')
                psf[np.where(cand)[0][0]] = True
            if verbose:
                print('Final selection: {0}'.format(np.where(psf)[0]))
        if verbose:
            print('  Selected {0} final PSF candidate(s).'.format(np.count_nonzero(psf)))
        for k, (fit, yslice, xslice) in enumerate(results):
            fit['psf'] = bool(psf[k])
        return results, psf

    def get_psf_profile(self, stamps, results, threshold=0.01, verbose=False):
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
        if threshold is not None:
            # Use interoplation to replace non-informative pixels.
            # This is useful for measurements that do not the ivar weights.
            noninformative = (Wsum <= threshold * np.median(Wsum))
            if np.any(noninformative):
                # Interpolate neighboring pixels.
                K = np.identity(3, dtype=np.float32)
                K[1, 1] = 0.
                WPconv = scipy.signal.convolve(Wsum * P, K, mode='same')
                Wconv = scipy.signal.convolve(Wsum, K, mode='same')
                P[noninformative] = np.divide(
                    WPconv[noninformative], Wconv[noninformative],
                    out=np.zeros(np.count_nonzero(noninformative)), where=Wconv[noninformative] > 0)
                Wsum[noninformative] = Wconv[noninformative]
            if verbose:
                print('Interpolated {0} non-informative pixels.'.format(np.count_nonzero(noninformative)))
        return P, Wsum

    def calculate_fwhm(self, profile, fiberfrac):
        """ Tabulate the circularized 1D PSF profile and calculate its FWHM.
        """
        # Locate the center of the profile.
        iy, ix = np.unravel_index(np.argmax(fiberfrac), fiberfrac.shape)
        xc = self.xyoffset[ix]
        yc = self.xyoffset[iy]
        # Calculate the radius of each pixel in arcsecs relative to this center.
        rangle = np.hypot((self.xgrid - xc) * self.pixel_size_um / self.plate_scales[0],
                          (self.ygrid - yc) * self.pixel_size_um / self.plate_scales[1]).reshape(-1)
        # Fill ivar-weighted histograms of flux versus angular radius.
        P, W = profile
        WZ, _ = np.histogram(rangle, bins=self.angbins, weights=(P * W).reshape(-1))
        W, _ = np.histogram(rangle, bins=self.angbins, weights=W.reshape(-1))
        # Calculate the circularized profile, normalized to 1 at (xc, yc).
        Z = np.divide(WZ, W, out=np.zeros_like(W), where=W > 0)
        Z /= Z[0]
        # Find the first bin where Z <= 0.5.
        k = np.argmax(Z <= 0.5)
        # Use linear interpolation over this bin to estimate FWHM.
        s = (0.5 - Z[k]) / (Z[k + 1] - Z[k])
        rangmid = self.profile_tab['rang']
        fwhm = 2 * ((1 - s) * rangmid[k] + s * rangmid[k + 1])
        return fwhm, Z, xc, yc

    def calculate_fiberfrac(self, profile):
        """ Tabulate the fiber acceptance fraction on a grid of centroid offsets.
        """
        # We only use the stacked profile for now.  Should revisit this to better
        # handle cases where some pixels are masked (W = 0) in the stack.
        P, W = profile
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
        return fiberfrac


class GuideCameraResults(object):
    """Container for guide camera analysis results.
    """
    def __init__(self, stamps, results, profile, fiberfrac, profile_tab, meta):
        self.stamps = stamps
        self.results = results
        self.profile = profile
        self.fiberfrac = fiberfrac
        self.profile_tab = profile_tab
        self.meta = meta

    def print(self):
        rsize = self.meta['SSIZE'] // 2
        print('FWHM = {0:.2f}" FIBERFRAC = {1:.3f}'.format(self.meta['FWHM'], self.meta['FFRAC']))
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
            # Write the fiberfraction values.
            hdus.write(self.fiberfrac, extname='FFRAC')
            # Write the tabulated data.
            hdus.write(self.profile_tab, extname='PROFTAB')

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
            fiberfrac = hdus['FFRAC'].read().copy()
            # Read the tabulated data.
            profile_tab = hdus['PROFTAB'].read().copy()
            return GuideCameraResults(stamps, results, profile, fiberfrac, profile_tab, meta)
