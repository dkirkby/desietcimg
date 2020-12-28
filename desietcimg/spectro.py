"""Utility functions for accessing DESI spectra for ETC calculations.
"""
from pathlib import Path
from contextlib import contextmanager

import numpy as np

import scipy.ndimage

import fitsio

try:
    import requests
except ImportError:
    # We will flag this later if it matters.
    pass


class DESIRoot(object):

    def __init__(self, localpath='/global/cfs/cdirs/desi', url='https://data.desi.lbl.gov/desi/', http_fallback=True):
        if Path(localpath).exists():
            self.mode = 'local'
        elif http_fallback:
            try:
                import requests
                self.mode = 'remote'
            except ImportError:
                raise RuntimeError('The requests package is not installed.')
        else:
            raise RuntimeError('Unable to access DESI files locally and http_fallback is False.')

    @contextmanager
    def openfits(name):
        if self.mode == 'local':
            hdus = fitsio.FITS(str(name))
            try:
                yield hdus
            finally:
                hdus.close()
        else:
            raise RuntimeError('openfits: remote mode not yet supported.')
