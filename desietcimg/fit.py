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
        # Pack the fit input and results into a recarray.
        self.data = np.empty(len(self.ydata),
            dtype=[('xpix', np.float32), ('ydata', np.float32), ('yfit', np.float32)])
        self.data['xpix'] = 0.5 * (self.xedge[1:] + self.xedge[:-1])
        self.data['ydata'] = self.ydata
        if result.success:
            self.data['yfit'] = self.predict(*result.x)
            return (result,) + tuple(result.x)
        else:
            self.data['yfit'] = self.predict(*theta0)
            return (result,) + tuple(theta0)


class FlatFieldFitter(object):
    """Fit flat field illumination to an interpolating polynomial.

    The interpolation order will be bicubic if there are sufficient
    knots, or otherwise downgraded to quadratic or linear.

    Parameters
    ----------
    shape : tuple
        Tuple (ny, nx) specifying the dimensions of the images to fit.
    downsampling : int
        Use a value > 1 to perform the fit to downsampled images
        with predictions integrated over each downsampled block.
    knots : tuple
        Number of knots (nyk, nxk) to use along the y- and x-axes.
        Knots are equally spaced between (and including) the corners
        of the image. For equal spacing along both axes, you should
        have (nyk - 1) / (nxk - 1) = ny / nx. At least 4 knots are
        required for cubic interpolation along each axis.
    """
    def __init__(self, shape, downsampling=32, knots=(5, 7)):
        (ny, nx) = self.shape = shape
        (nyk, nxk) = self.knots = knots
        # Space knots equally between the image corners.
        self.x_knot = np.linspace(-0.5, nx - 0.5, nxk)
        self.y_knot = np.linspace(-0.5, ny - 0.5, nyk)
        # Use bicubic interpolation if possible.
        self.kx = max(nxk - 1, 3)
        self.ky = max(nyk - 1, 3)
        self.downsampling = downsampling
        # Evaluate the interpolation at each pixel center.
        self.xc = np.arange(nx)
        self.yc = np.arange(ny)
        # Integrate the interpolation over downsampled blocks.
        self.xd = np.arange((nx // downsampling) + 1) * downsampling - 0.5
        self.yd = np.arange((ny // downsampling) + 1) * downsampling - 0.5
        self.nxd = len(self.xd) - 1
        self.nyd = len(self.yd) - 1

    def predict(self, theta, downsampled=True):
        """Predict the bias subtracted flat field signal in ADU.

        Parameters
        ----------
        theta : array
            2D array of shape self.knots containing the illumination
            values at each knot, in bias subtracted ADUs.
        downsampled : bool
            When False, calculate a prediction at each pixel center.
            This is equivalent to downsampling == 1 and temporarily
            overrides the actual value of downsampling.

        Returns
        -------
        array
            2D array of predicted values. If downsampling == 1, values are
            calculated at the center of each pixel. If downsampling > 1,
            values are pixel means integrated over each downsampled block.
        """
        theta = self.yscale * theta + self.y0
        spline = scipy.interpolate.RectBivariateSpline(
            self.y_knot, self.x_knot, theta.reshape(self.knots))
        if not downsampled or self.downsampling == 1:
            return spline(self.yc, self.xc, grid=True)
        else:
            pred = np.empty((self.nyd, self.nxd))
            for iy in range(self.nyd):
                for ix in range(self.nxd):
                    pred[iy, ix] = spline.integral(
                        self.yd[iy], self.yd[iy + 1], self.xd[ix], self.xd[ix + 1])
            # Normalize to average per pixel within each downsampled block.
            return pred / self.downsampling ** 2

    def nll(self, theta, ydata, valid, rdnoise, gain):
        """Calculate the -logL of the data given parameters theta.

        The variance in ADU ** 2 of each observed pixel value is assumed
        to be rdnoise ** 2 + gain * prediction, where rdnoise and the
        prediction are in ADU.
        
        Parameters
        ----------
        theta : array
            2D array of shape self.knots containing the illumination
            values at each knot, in bias subtracted ADUs.
        ydata : array
            2D array of bias subtracted observed pixel values. When
            downsampling > 1, values should be averages over the valid pixels
            in each downsampled block.
        valid : array
            2D array of 0 and 1 values where 1 indicates a valid pixel to
            use in the calculation.
        rdnoise : float
            Read noise in ADU to use in the noise model.
        gain : float
            Inverse gain in e/ADU to use in the noise model.
        """
        ypred = self.predict(theta)
        ivar = valid / (rdnoise ** 2 + gain * ypred)
        result = 0.5 * np.sum(ivar * (ydata - ypred) ** 2)
        return result

    def fit(self, raw, mask, bias, rdnoise, gain, navg=32, verbose=True):
        """Fit raw image data to a smooth flat field illumination model.

        Parameters
        ----------
        raw : array
            2D array of (ny, nx) raw pixel values in ADU.
        mask : array
            2D array of (ny, nx) pixel mask values, where a value of
            zero indicates a valid pixel to use in the fit.
        bias : array or float
            Bias value in ADU to subtract from each raw pixel value.
            Can be a constant or 2D array of shape (ny, nx).
        rdnoise : float
            Read noise in ADU to use in the noise model.
        gain : float
            Inverse gain in e/ADU to use in the noise model.
        navg : int
            Initial knot parameters for the fit are averages over a
            block of size 2 * navg + 1 pixels centered on each knot.
        verbose : bool
            Report any fit issues when True.

        Returns
        -------
        tuple
            Tuple (fit, ypred) where fit is a scipy.optimize.OptimizeResult
            and ypred is a 2D array of (ny, nx) predicted bias-subtracted
            pixel values using the best fit parameters, when fit.success is
            True or otherwise the initial fit parameters.
        """
        (ny, nx) = self.shape
        if raw.shape != self.shape or mask.shape != self.shape:
            raise ValueError('Input arrays have the wrong shape.')

        # Calculate the bias subtracted data.
        ydata = (raw - bias).astype(np.float64)
        valid = (mask == 0).astype(np.float64)

        # Calculate coefficients for scaling the fit parameters to O(1).
        self.y0 = np.median(ydata[valid > 0])
        self.yscale = np.std(ydata[valid > 0])

        # Calculate the initial fit parameters as mean values near each knot.
        # The default is y0 when no valid pixels are available for averaging.
        theta0 = np.full(self.knots, self.y0)
        for i, y in enumerate(self.y_knot):
            iy = int(round(y))
            ylo, yhi = max(0, iy - navg), min(ny, iy + navg + 1)
            for j, x in enumerate(self.x_knot):
                ix = int(round(x))
                xlo, xhi = max(0, ix - navg), min(nx, ix + navg + 1)
                nvalid = np.sum(valid[ylo:yhi, xlo:xhi])
                if nvalid > 0:
                    theta0[i, j] = np.sum(ydata[ylo:yhi, xlo:xhi]) / nvalid                    
        # Normalize the fit parameters.
        theta0 = (theta0 - self.y0) / self.yscale

        # Downsample the data if requested, using only valid pixels to
        # calculated the downsampled mean over each block.
        if self.downsampling > 1:
            DW = desietcimg.util.downsample(
                ydata * valid, self.downsampling, np.sum, allow_trim=True)
            W = desietcimg.util.downsample(
                valid, self.downsampling, np.sum, allow_trim=True)
            ydata = np.divide(DW, W, out=np.zeros_like(DW), where=W>0)
            valid = (W > 0).astype(np.float64)

        # Try the fit with BFGS first.
        result = scipy.optimize.minimize(
            self.nll, theta0, args=(ydata, valid, rdnoise, gain),
            method='BFGS', jac='2-point', options=dict(
                maxiter=10000, gtol=1e-2, disp=False))

        if not result.success:
            if verbose:
                print('BFGS fit failed, so will retry with Nelder-Mead.')
                print(result.message)
            # Try again from the BFGS result using Nelder-Mead.
            theta0 = result.x
            result = scipy.optimize.minimize(
                self.nll, theta0, args=(ydata, valid, rdnoise, gain),
                method='Nelder-Mead', options=dict(
                    maxiter=10000, xatol=1e-3, fatol=1e-3, disp=False))
            if not result.success and verbose:
                print('Nelder-Mead also failed.')
                print(result.message)

        theta = result.x if result.success else theta0
        yfit = self.predict(theta, downsampled=False)
        return result, yfit
