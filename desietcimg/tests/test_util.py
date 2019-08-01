import unittest
import numpy as np
from desietcimg.util import *


class TestUtil(unittest.TestCase):

    def test_downsample(self):
        self.assertTrue(np.array_equal(
            downsample(np.ones((9, 12)), 3, allow_trim=False),
            np.full((3, 4), 9.)))
        self.assertTrue(np.array_equal(
            downsample(np.ones((11, 10)), 2, allow_trim=True),
            np.full((5, 5), 4.)))
        with self.assertRaises(ValueError):
            downsample(np.ones((10, 11)), 2, allow_trim=False)

    def test_make_template(self, r=6):
        profile = lambda x, y: x ** 2 + y ** 2 < r ** 2
        T1 = make_template(15, profile, normalized=True)
        self.assertTrue(np.allclose(T1.sum(), 1.))
        T2 = make_template(15, profile, oversampling=64, normalized=False)
        self.assertTrue(np.allclose(T2.sum(), np.pi * r ** 2, rtol=2e-4))
        T3 = make_template(15, profile, oversampling=64, dy=1, normalized=False)
        self.assertTrue(np.allclose(T2[:-1], T3[1:]))

    def test_Convolutions(self, nx=12, ny=9, ksize=5):
        source = np.arange(nx * ny).reshape(ny, nx)
        kernel = np.arange(ksize ** 2).reshape(ksize, ksize)
        C = Convolutions([source], kernel)
        k = nx * ny
        for x1 in range(0, nx - 1):
            for x2 in range(x1 + 1, nx):
                for y1 in range(0, ny - 1):
                    for y2 in range(y1 + 1, ny):
                        C.set_source(slice(y1, y2), slice(x1, x2), k)
                        xcheck = scipy.signal.convolve(C.sources[0], C.kernel, mode='same')
                        assert np.allclose(C.convolved[0], xcheck)
                        k += 1
        for x in range(nx):
            for y in range(ny):
                C.set_source(y, x, k)
                xcheck = scipy.signal.convolve(C.sources[0], C.kernel, mode='same')
                assert np.allclose(C.convolved[0], xcheck)
                k += 1

    def test_prepare(self, size=100, nelec=1e6, bias=1e3, bgrms=20., invgain=10, seed=123):
        gauss = lambda x, y: np.exp(-0.5 * (x ** 2 + y ** 2) / (0.03 * size) ** 2)
        signal_elec = nelec * make_template(size, gauss, oversampling=2, normalized=True)
        signal_adu = signal_elec / invgain
        var_adu = signal_elec / invgain ** 2
        var_tot = var_adu + bgrms ** 2
        rng = np.random.RandomState(seed=seed)
        D = rng.normal(loc=signal_adu + bias, scale=np.sqrt(var_tot))
        D, W = prepare(D, invgain=invgain)
        clipped, _, _ = scipy.stats.sigmaclip(D)
        self.assertTrue(np.abs(np.mean(clipped)) < 1e-4)
        rho = np.corrcoef(var_tot.reshape(-1), 1 / W.reshape(-1))[0, 1]
        self.assertTrue(rho > 0.8)


if __name__ == '__main__':
    unittest.main()
