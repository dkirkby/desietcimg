"""Utility functions for accessing DESI spectra for ETC calculations.
"""
import numpy as np

import scipy.ndimage

import fitsio


wmin, wmax, wdelta = 3600, 9824, 0.8
fullwave = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice = {'b': slice(0, 2751), 'r': slice(2700, 5026), 'z': slice(4900, 7781)}


class Spectrum(object):
    def __init__(self, stype, flux=None, ivar=None, mask=None):
        assert stype == 'full' or stype in cslice, 'invalid stype'
        self.stype = stype
        self.wave = fullwave[cslice[stype]] if stype in cslice else fullwave
        if flux is None and ivar is None:
            self._flux = np.zeros(len(self.wave))
            self.ivar = np.zeros(len(self.wave))
        elif flux is not None and ivar is not None:
            self._flux = np.asarray(flux)
            self.ivar = np.asarray(ivar)
            assert self.ivar.shape == self._flux.shape, 'flux and ivar have different shapes.'
        else:
            raise ValueError('flux and ivar must both be specified.')
        if mask is None:
            self.mask = np.zeros_like(self._flux, bool)
        else:
            self.mask = np.asarray(mask)
            assert self.mask.shape == self._flux.shape, 'flux and mask have different shapes.'
    def copy(self):
        return Spectrum(self.stype, self.flux.copy(), self.ivar.copy(), self.mask.copy())
    def __itruediv__(self, factor):
        np.divide(self.flux, factor, out=self._flux, where=factor != 0)
        self.ivar *= factor ** 2
        return self
    def __truediv__(self, factor):
        result = self.copy()
        result /= factor
        return result
    @property
    def flux(self):
        return self._flux


class CoAdd(Spectrum):
    def __init__(self, stype):
        super(CoAdd, self).__init__(stype)
        self._weighted_flux_sum = np.zeros(len(self.wave))
        self._finalized = False
    def __iadd__(self, other):
        if other.stype == self.stype:
            self_slice = slice(None, None)
        elif self.stype == 'full':
            self_slice = cslice[other.stype]
        else:
            raise ValueError(f'Cannot add "{other.stype}" to "{self.stype}".')
        self._weighted_flux_sum[self_slice] += other.ivar * other.flux
        self.ivar[self_slice] += other.ivar
        self._finalized = False
        return self
    @property
    def flux(self):
        if not self._finalized:
            np.divide(self._weighted_flux_sum, self.ivar, out=self._flux, where=self.ivar > 0)
            self._finalized = True
        return self._flux
