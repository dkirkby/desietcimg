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


if __name__ == '__main__':
    unittest.main()
