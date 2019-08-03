import numpy as np

import desietcimg.util


def moffat_profile(x, y, fwhm, sx=1, sy=1, beta=3.5):
    """Moffat profile to model nominal PSF
    """
    r0 = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    r = np.sqrt((x / sx) ** 2 + (y / sy) ** 2)
    return (1 + (r / r0) ** 2) ** -beta


