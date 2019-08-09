import functools

import numpy as np

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
        # Initialize fitter.
        self.fitter = desietcimg.fit.GaussFitter(stamp_size)

    def detect_sources(self, D, W=None, nsrc_max=12, chisq_max=150., min_central=18,
                       size_min=3.5, ratio_min=0.7, snr_min=50., cdist_max=3.):
        """Detect PSF-like sources in an image.

        Parameters
        ----------
        D : array
            2D array of image pixel values with shape (ny, nx).
        W : array or None
            2D array of correponsding inverse-variance weights with shape (ny, nx).
            When None, this array will be estimated from D.
        nsrc_max : int
            Maximum number of sources to detect.
        chisq_max : float
            Cut on chisq value passed to :func:`desietcimg.util.mask_defects` for
            each candidate stamp.
        min_central : int
            Minimum number of unmasked pixels required in the central 5x5 region.
            Used to reject saturated stars.
        ...

        Returns
        -------
        ...
        """
        D, W = desietcimg.util.prepare(D, W)
        # Mask the most obvious defects in the whole image with a very loose chisq cut.
        W, nmasked = desietcimg.util.mask_defects(D, W, 1e4, inplace=True)
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
        stamps = []
        params = []
        AX = desietcimg.plot.Axes(nsrc_max)
        while len(stamps) < nsrc_max:
            ax = AX.axes[len(stamps)]

            # Find the largest filtered value in the inset region and its indices [iy, ix].
            iy, ix = np.unravel_index(np.argmax(inset), (ny - 2 * h, nx - 2 * h))
            fmax = inset[iy, ix]
            
            # Extract the stamp centered on [iy, ix] in the full (non-inset) image.
            xlo, ylo = ix, iy
            xhi, yhi = ix + ss, iy + ss
            stamp = D[ylo:yhi, xlo:xhi].copy()
            ivar = W[ylo:yhi, xlo:xhi].copy()
            
            # Find the largest filtered value outside of this stamp.
            save = filtered[ylo:yhi, xlo:xhi].copy()
            filtered[ylo:yhi, xlo:xhi] = fmin
            f2nd = np.max(inset)
            filtered[ylo:yhi, xlo:xhi] = save

            # Mask pixel defects in this stamp.
            ivar, nmasked = desietcimg.util.mask_defects(stamp, ivar, chisq_max, inplace=True)
            
            # Update the WD and W convolutions with the new ivar.
            CWD.set_source(slice(ylo, yhi), slice(xlo, xhi), stamp * ivar)
            changed = CW.set_source(slice(ylo, yhi), slice(xlo, xhi), ivar)
            # Calculate the updated filtered array.
            filtered[changed] = 0
            filtered[changed] = np.divide(
                WDf[changed], Wf[changed], out=filtered[changed], where=Wf[changed] > 0)

            # Calculate the change in the filtered value after the masking.
            # We can assume that the denominator is non-zero here.
            fnew = np.sum(stamp * ivar * self.PSF0) / np.sum(ivar * self.PSF0)

            if fnew < f2nd:
                # This stamp had a artificially high filtered value due to pixel
                # defects so skip it now.  It might still come back at a lower
                # value after masking.
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
                # useful. This is normal for saturated stars.
                continue

            '''
            # Stamps are sometimes selected on the wings of a previously selected
            # bright star.  To detect this condition, calculate the average pixel flux
            # in 5x5 windows at N, E, S, W edges and compare with a central window.
            central_avg = (
                np.sum(ivar[c_slice, c_slice] * stamp[c_slice, c_slice]) /
                np.sum(ivar[c_slice, c_slice]))
            lo_slice = slice(0, 2 * nwindow + 1)
            hi_slice = slice(ss - 2 * nwindow - 1, ss)
            wing_condition = False
            for x_slice, y_slice in ((lo_slice, None), (hi_slice, None), (None, lo_slice), (None, hi_slice)):
                ivar_sum = np.sum(ivar[y_slice, x_slice])
                if ivar_sum == 0:
                    continue
                edge_avg = np.sum(ivar[y_slice, x_slice] * stamp[y_slice, x_slice]) / ivar_sum
                print(x_slice, y_slice, edge_avg, central_avg)
                if edge_avg > central_avg:
                    wing_condition = True
                    break
            if wing_condition:
                continue
            '''

            # Calculate the stamp's centroid (w/o centered PSF weights)
            clipped = np.maximum(0., stamp)
            M0 = np.sum(clipped * ivar)
            Mx = np.sum(self.xgrid * clipped * ivar) / M0
            My = np.sum(self.ygrid * clipped * ivar) / M0
            
            # Calculate the centroid distance from the stamp center.
            cdist = np.sqrt(Mx ** 2 + My ** 2)

            # Ignore stamps whose centroid is not centered, which probably indicates
            # we have found the wing of a previously found bright star.
            if cdist > cdist_max:
                continue

            '''
            # Calculate the stamp's template-weighted second moments
            # relative to the stamp center.
            M0 = np.sum(self.PSF0 * clipped * ivar)
            Mxx = np.sum(self.PSFxx * clipped * ivar) / M0
            Mxy = np.sum(self.PSFxy * clipped * ivar) / M0
            Myy = np.sum(self.PSFyy * clipped * ivar) / M0
   
            # Calculate the determinant size.
            det = Mxx * Myy - Mxy ** 2
            size = det ** 0.25 if det > 0 else 0.
            
            # Calculate the minor / major axis ratio.
            trace = Mxx + Myy
            diff = np.sqrt((Mxx - Myy) ** 2 + 4 * Mxy ** 2)
            major = np.sqrt(0.5 * (trace + diff))
            minor = np.sqrt(0.5 * (trace - diff))
            ratio = minor / major
            #print(trace, diff, major, minor, ratio)
            
            # Calculate the SNR2 as the weighted mean of the pixel SNR2 value.
            SNR = np.sum(stamp * self.PSF0 * ivar) / np.sqrt(np.sum(self.PSF0 ** 2 * ivar))

            # Is this stamp sufficiently PSF like?
            keep = (SNR > snr_min) and (size > size_min) and (ratio > ratio_min)
            '''

            desietcimg.plot.plot_image(stamp, ivar, ax=ax)

            # Fit a single Gaussian + constant background to this stamp.
            results = self.fitter.fit(stamp, ivar)

            if results.success or results.status == 2:
                ls = '-' if results.success else ':'
                desietcimg.plot.draw_ellipse(
                    ax, results.p['x0'], results.p['y0'], results.p['s'], results.p['g1'], results.p['g2'], ls=ls)
                '''
                label = f'$\\nu$ {SNR:.1f} $\\sigma$ {size:.1f} $r$ {ratio:.2f}'
                ax.text(0.5, 0.95, label, horizontalalignment='center', verticalalignment='center',
                        fontsize=22, fontweight='bold', color='w' if keep else 'k', transform=ax.transAxes)
                '''
            stamps.append((stamp, ivar))
            params.append((results, slice(ylo, yhi), slice(xlo, xhi)))
        self.stamps = stamps
        self.params = params
