import json
import collections
import functools

import numpy as np

import scipy.special
import scipy.signal
import scipy.stats

import desietcimg.util


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
        # Calculate the ivar-weighted image.
        WD = np.array(W * D, np.float32)
        # Convolve the image with each filter.
        CWD = desietcimg.util.Convolutions([WD], self.PSF0)
        WD = CWD.sources[0]
        WDf = CWD.convolved[0]
        CW = desietcimg.util.Convolutions([W], self.PSF0)
        Wf = CW.convolved[0]
        filtered = np.divide(WDf, Wf, out=np.zeros_like(W), where=Wf > 0)
        stamps = []
        params = []
        AX = desietcimg.plot.Axes(nsrc_max)
        while len(stamps) < nsrc_max:
            ax = AX.axes[len(stamps)]

            # Find the next largest filtered pixel.
            iy, ix = np.unravel_index(np.argmax(filtered), (ny, nx))
            xlo, ylo = max(0, ix - h), max(0, iy - h)
            xhi, yhi = min(nx, ix + h + 1), min(ny, iy + h + 1)

            # Redo the filtering with the data in this stamp set to zero (for next time).
            changed = CWD.set_source(slice(ylo, yhi), slice(xlo, xhi), 0)
            filtered[changed] = 0
            filtered[changed] = np.divide(
                WDf[changed], Wf[changed], out=filtered[changed], where=Wf[changed] > 0)
                        
            # Ignore stamps that are not fully contained.
            if ix < h or iy < h or ix + h >= nx or iy + h >= ny:
                continue
            
            stamp = D[ylo:yhi, xlo:xhi].copy()
            ivar = W[ylo:yhi, xlo:xhi].copy()
            
            #plot_image(stamp, ivar, ax=ax)
            
            # Do a second pass of defect masking with tighter cuts.
            ivar, nmasked = desietcimg.util.mask_defects(stamp, ivar, 100., verbose=False)
            
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
            ##SNR = np.sqrt(max(0., np.sum(self.PSF0 * ivar * stamp ** 2)))
            SNR = np.sum(stamp * self.PSF0 * ivar) / np.sqrt(np.sum(self.PSF0 ** 2 * ivar))

            # Is this stamp sufficiently PSF like?
            keep = (SNR > snr_min) and (size > size_min) and (ratio > ratio_min)

            desietcimg.plot.plot_image(stamp, ivar, cov=[[Mxx, Mxy], [Mxy, Myy]], ax=ax)
            
            label = f'$\\nu$ {SNR:.1f} $\\sigma$ {size:.1f} $r$ {ratio:.2f}'
            ax.text(0.5, 0.95, label, horizontalalignment='center', verticalalignment='center',
                    fontsize=22, fontweight='bold', color='w' if keep else 'k', transform=ax.transAxes)
            
            stamps.append((stamp, ivar))
            params.append((SNR, size, ratio))
