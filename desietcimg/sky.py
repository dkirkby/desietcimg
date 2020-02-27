import collections

import numpy as np

import scipy.optimize
import scipy.stats

import desietcimg.util


class BGFitter(object):
    """Fit a histogram of pixel values to a single Gaussian noise model.
    """
    def __init__(self, optimize_args={}):
        # Initialize the args sent to scipy.optimize.minimize()
        self.kwargs = dict(
            method='Nelder-Mead',
            options=dict(maxiter=10000, xatol=1e-4, fatol=1e-4, disp=False))
        self.kwargs.update(optimize_args)

    def predict(self, ntot, mu, std):
        """Predict data with specified parameters and bin edges.
        """
        z = scipy.special.erf((self.xedge - mu) / (np.sqrt(2) * std))
        return 0.5 * ntot * np.diff(z)

    def nll(self, theta):
        """Objective function for minimization, calculates -logL.
        """
        ypred = self.predict(*theta)
        # Use the model to predict the Gaussian inverse variances.
        yvar = np.maximum(1, ypred)
        return 0.5 * np.sum((self.ydata - ypred) ** 2 / yvar)

    def fit(self, data, nbins=30, maxpct=90):
        data = data.reshape(-1)
        clipped, lo, hi = scipy.stats.sigmaclip(data, low=3, high=2)
        if np.issubdtype(data.dtype, np.integer):
            # Align bins to integer boundaries.
            lo, hi = np.floor(lo) - 0.5, np.ceil(hi) + 0.5
            binsize = np.round(np.ceil(hi - lo) / nbins)
            nbins = np.ceil((hi - lo) / binsize)
            self.xedge = lo + np.arange(nbins) * binsize
        else:
            self.xedge = np.linspace(lo, hi, nbins)
        self.ydata, _ = np.histogram(data, bins=self.xedge)
        xc = 0.5 * (self.xedge[1:] + self.xedge[:-1])
        theta0 = np.array([1, np.mean(clipped), np.std(clipped)])
        y0 = self.predict(*theta0)
        imode = np.argmax(y0)
        theta0[0] = self.ydata[imode] / y0[imode]
        y0 = self.predict(*theta0)
        result = scipy.optimize.minimize(self.nll, theta0, **self.kwargs)
        if result.success:
            theta = result.x
        else:
            theta = theta0
        return theta
