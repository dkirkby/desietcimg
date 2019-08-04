import functools

import numpy as np

import desietcimg.util


def moffat_profile(x, y, fwhm, sx=1, sy=1, beta=3.5):
    r0 = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    r = np.sqrt((x / sx) ** 2 + (y / sy) ** 2)
    return (1 + (r / r0) ** 2) ** -beta


class GuideCameraAnalysis(object):
    """
    """
    def __init__(self, stamp_size=75, pixel_size_um=9, plate_scales=(70., 76.), match_fwhm_arcsec=1.1):
        assert stamp_size % 2 == 1
        self.rsize = stamp_size // 2
        self.stamp_size = stamp_size
        # Build a nominal PSF model for detection matching.
        profile = functools.partial(moffat_profile, fwhm=match_fwhm_arcsec,
                                    sx=plate_scales[0] / pixel_size_um, sy=plate_scales[1] / pixel_size_um)
        self.PSF0 = desietcimg.util.make_template(stamp_size, profile, normalized=True)
        # Precompute PSF-weighted images for calculating second moments.
        dxy = np.arange(stamp_size) - 0.5 * (stamp_size - 1)
        self.xgrid, self.ygrid = np.meshgrid(dxy, dxy, sparse=False)
        self.PSFxx = self.PSF0 * self.xgrid ** 2
        self.PSFxy = self.PSF0 * self.xgrid * self.ygrid
        self.PSFyy = self.PSF0 * self.ygrid ** 2

    def detect_sources(self, D, W=None, nsrc_max=12, chisq_max=1e4,
                       size_min=3.5, ratio_min=0.7, snr_min=50., cdist_max=3.):
        D, W = desietcimg.util.prepare(D, W)
        W, nmasked = desietcimg.util.mask_defects(D, W, chisq_max, verbose=False)
        print('masked', nmasked, 'defects')
        ny, nx = D.shape
        h = self.rsize
        ss = self.stamp_size
        # Calculate the ivar-weighted image.
        WD = np.array(W * D, np.float32)
        # Convolve the image with each filter.
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
            ivar, nmasked = desietcimg.util.mask_defects(stamp, ivar, 100., verbose=False)
            
            # Update the WD and W convolutions with the new ivar.
            CWD.set_source(slice(ylo, yhi), slice(xlo, xhi), stamp * ivar)
            changed = CW.set_source(slice(ylo, yhi), slice(xlo, xhi), ivar)
            # Calculate the updated filtered array.
            filtered[changed] = 0
            filtered[changed] = np.divide(
                WDf[changed], Wf[changed], out=filtered[changed], where=Wf[changed] > 0)

            # Calculate the change in the filtered value after the masking.
            # Can assume that the denominator is non-zero here.
            fnew = np.sum(stamp * ivar * self.PSF0) / np.sum(ivar * self.PSF0)

            if fnew < f2nd:
                # This is no longer the next stamp to consider.
                continue

            # Redo the filtering with the data in this stamp set to zero.
            changed = CWD.set_source(slice(ylo, yhi), slice(xlo, xhi), 0)
            filtered[changed] = 0
            filtered[changed] = np.divide(
                WDf[changed], Wf[changed], out=filtered[changed], where=Wf[changed] > 0)

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

            desietcimg.plot.plot_image(stamp, ivar, cov=[[Mxx, Mxy], [Mxy, Myy]], ax=ax)            
            label = f'$\\nu$ {SNR:.1f} $\\sigma$ {size:.1f} $r$ {ratio:.2f}'
            ax.text(0.5, 0.95, label, horizontalalignment='center', verticalalignment='center',
                    fontsize=22, fontweight='bold', color='w' if keep else 'k', transform=ax.transAxes)
            
            stamps.append((stamp, ivar))
            params.append((SNR, size, ratio))
