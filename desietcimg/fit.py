"""Fitting methods.
"""
import warnings

import numpy as np

import scipy.optimize


class GaussFitter(object):
    """Fit a single 2D Gaussian + constant background to a square image.
    
    Parameters
    ----------
    stamp_size : int
        Size of square stamps to fit.
    bgmargin : int
        A margin of this size in pixels is used for an initial estimate
        of the background level. Pixels inside this margin are used to
        estimate the total signal.
    r0sig : float
        Hyperparameter specifying the sigma in pixels for a Gaussian
        prior in the centroid offset r0 from the stamp center.
    gsig : float
        Hyperparmeter specifying the sigma for a Gaussian prior on the
        ellipticity magnitude g = sqrt(g1 ** 2 + g2 ** 2).
    optimize_args : dict
        Dictionary of arguments to be passed to scipy.optimize.minimize().
    """
    def __init__(self, stamp_size, bgmargin=8, r0sig=5, gsig=5, optimize_args={}): 
        # Initialize the args sent to scipy.optimize.minimize()
        self.kwargs = dict(method='BFGS', jac='2-point',
            options=dict(maxiter=10000, gtol=1e-2, disp=False))
        self.kwargs.update(optimize_args)
        # Tabulate the pixel grid to use.
        self.xy = np.arange(stamp_size) - 0.5 * (stamp_size - 1)
        # Save the prior hyperparameters.
        self.r0sig = r0sig
        self.gsig = gsig
        # Build masks for the signal and background estimation regions.
        self.bgmask = np.zeros((stamp_size, stamp_size), bool)
        self.bgmask[:bgmargin] = True
        self.bgmask[-bgmargin:] = True
        self.bgmask[:, :bgmargin] = True
        self.bgmask[:, -bgmargin:] = True
        self.sigmask = ~self.bgmask
        self.nsigmask = np.count_nonzero(self.sigmask)
    
    def predict(self, b, logf, x0, y0, logs, g1, g2):
        """Predict data with specified parameters.
        """
        dx = self.xy - x0
        dy = (self.xy - y0).reshape(-1, 1)
        gsq = g1 ** 2 + g2 ** 2
        s = np.exp(logs)
        norm = s ** 2 * (1 - gsq)
        Qinvxx = (1 + gsq - 2 * g1) / norm
        Qinvxy = -2 * g2 / norm
        Qinvyy = (1 + gsq + 2 * g1) / norm
        arg = Qinvxx * dx ** 2 + 2 * Qinvxy * dx * dy + Qinvyy * dy ** 2
        signal = np.exp(-0.5 * arg)
        return b + np.exp(logf) * signal / signal.sum()
    
    def nlprior(self, theta):
        """Evaluate -log P(theta)
        """
        _, _, x0, y0, _, g1, g2 = theta
        r0sq = x0 ** 2 + y0 ** 2
        gsq = g1 ** 2 + g2 ** 2
        return 0.5 * r0sq / self.r0sig ** 2  + 0.5 * gsq / self.gsig ** 2
    
    def nlpost(self, theta, D, W):
        """Evaluate -log P(theta | D, W)
        """
        Dpred = self.predict(*theta)
        nllike = 0.5 * np.sum(W * (D - Dpred) ** 2)
        return nllike + self.nlprior(theta)
    
    def fit(self, D, W, s0=5):
        """Fit an image D with inverse variance weights W.
        
        Parameters
        ----------
        D : array
            Square 2D array of pixel values.
        W : array
            Square 2D array of pixel inverse variances. Must have same shape as D.
        s0 : float
            Initial guess at size parameter in pixels.
        
        Returns
        -------
        scipy.optimize.OptimizeResult
            Result of scipy.optimize.minimize call.
        """
        ny, nx = D.shape
        assert ny == nx
        assert D.shape == W.shape
        assert np.all(W >= 0)
        # Estimate the background and signal parameters.
        bg = np.sum(W * D * self.bgmask) / np.sum(W * self.bgmask)
        sig = np.sum(W * (D - bg) * self.sigmask) / np.sum(W * self.sigmask) * self.nsigmask
        # Build the vector of initial parameter values.
        self.theta0 = np.array([bg, np.log(sig), 0., 0., np.log(s0), 0., 0.])
        # Run the optimizer and return the result.
        try:
            # Silently ignore any warnings.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                results = scipy.optimize.minimize(self.nlpost, self.theta0, args=(D, W), **self.kwargs)
            if results.success:
                results.p = dict(zip(('b', 'f', 'x0', 'y0', 's', 'g1', 'g2'), results.x))
                results.p['f'] = np.exp(results.p['f'])
                results.p['s'] = np.exp(results.p['s'])
        except Exception as e:
            print(str(e))
            results = dict(success=False, message=str(e))
        return results
