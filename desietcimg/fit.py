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
        self.trace = False

    # Name each of our (transformed) parameters.
    pnames = ('b', 'f', 'x0', 'y0', 's', 'g1', 'g2')


    def transform(self, theta):
        """Undo the non-linear transforms that implement our parameter constraints:
        s > 0, f > 0, g1 ** 2 + g2 ** 2 < 1
        """
        b, logfr, x0, y0, logs, G1, G2 = theta
        s = np.exp(logs)
        f = self.f0 * np.exp(logfr)
        Gscale = np.sqrt(1 + G1 ** 2 + G2 ** 2)
        g1 = G1 / Gscale
        g2 = G2 / Gscale
        return b, f, x0, y0, s, g1, g2
    
    def predict(self, b, f, x0, y0, s, g1, g2):
        """Predict data with specified parameters.
        """
        # Calculate pixel coordinates relative to Gaussian center.
        dx = self.xy - x0
        dy = (self.xy - y0).reshape(-1, 1)
        # Calculate the 2nd-moment matrix normalization.
        gsq = g1 ** 2 + g2 ** 2
        norm = s ** 2 * (1 - gsq)
        Qinvxx = (1 + gsq - 2 * g1) / norm
        Qinvxy = -2 * g2 / norm
        Qinvyy = (1 + gsq + 2 * g1) / norm
        arg = Qinvxx * dx ** 2 + 2 * Qinvxy * dx * dy + Qinvyy * dy ** 2
        signal = np.exp(-0.5 * arg)
        # Combine normalized signal and background.
        return b + f * signal / signal.sum()

    def nlprior(self, b, f, x0, y0, s, g1, g2):
        """Evaluate -log P(theta)
        """
        r0sq = x0 ** 2 + y0 ** 2
        gsq = g1 ** 2 + g2 ** 2
        return 0.5 * r0sq / self.r0sig ** 2  + 0.5 * gsq / self.gsig ** 2

    def nlpost(self, theta, D, W):
        """Evaluate -log P(theta | D, W)
        """
        params = self.transform(theta)
        Dpred = self.predict(*params)
        nllike = 0.5 * np.sum(W * (D - Dpred) ** 2)
        result = nllike + self.nlprior(*params)
        if self.trace:
            print('[{0}]'.format(self.ncall), end='')
            for pname, pvalue in zip(self.pnames, params):
                print(' {0}={1:.3f}'.format(pname, pvalue), end='')
            print(' -> {0}'.format(result))
            self.ncall += 1
        return result

    def fit(self, D, W, s0=5, verbose=False, trace=False):
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
        # Initialize default (unsuccessful) result.
        result = dict(success=False, message='Never got started', status=-1, snr=-1)
        # Estimate the background and signal parameters.
        Wsum = np.sum(W * self.bgmask)
        bg = np.sum(W * D * self.bgmask) / Wsum if Wsum > 0 else 0
        Wsum = np.sum(W * self.sigmask)
        WDsum = np.sum(W * (D - bg) * self.sigmask)
        sig = WDsum / Wsum * self.nsigmask if (WDsum > 0 and Wsum > 0) else 1
        if verbose:
            print('Estimated: b={0:.2f} adu/pix s={1:.1f} adu'.format(bg, sig))
        # Build the vector of initial parameter values.
        self.f0 = sig
        self.theta0 = np.array([bg, 0., 0., 0., np.log(s0), 0., 0.])
        # Run the optimizer and return the result.
        try:
            self.ncall = 0
            self.trace = trace
            # Silently ignore any warnings.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fitresult = scipy.optimize.minimize(self.nlpost, self.theta0, args=(D, W), **self.kwargs)
                result['success'] = fitresult.success
                result['message'] = fitresult.message
                result['status'] = fitresult.status
            if fitresult.success:
                result.update(dict(zip(self.pnames, self.transform(fitresult.x))))
                # Render with the best-fit parameters.
                P = self.predict(*self.transform(fitresult.x))
                # Calculate the P-weighted signal-to-noise ratio.
                result['snr'] = np.sum(D * P * W) / np.sqrt(np.sum(P ** 2 * W))
        except Exception as e:
            print(str(e))
            result['message'] = str(e)
        return result


class CalibFitter(object):
    """Fit a histogram of pixel values to a single Gaussian noise model.
    """
    def __init__(self, optimize_args={}):
        # Initialize the args sent to scipy.optimize.minimize()
        self.kwargs = dict(
            method='Nelder-Mead',
            options=dict(maxiter=10000, xatol=1e-3, fatol=1e-3, disp=False))
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
    
    def fit(self, ilo, ihi, ydata, ntot0, mu0, std0):
        self.xedge = np.arange(ilo, ihi + 1) - 0.5
        self.ydata = np.asarray(ydata, np.float)
        theta0 = np.array([ntot0, mu0, std0], np.float)
        result = scipy.optimize.minimize(self.nll, theta0, **self.kwargs)
        if result.success:
            self.yfit = self.predict(*result.x)
            return (result,) + tuple(result.x)
        else:
            self.yfit = self.predict(*theta0)
            return (result,) + tuple(theta0)
