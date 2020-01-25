import numpy as np
import scipy.special


class GMMFit(object):

    def __init__(self, x1_edges, x2_edges):
        """
        Initialize a Gaussian mixture model fitter.

        Parameters
        ----------
        x1_edges : array
            1D array of n1+1 increasing values that specify the pixel edges
            along the x1 direction.
        x2_edges : array
            1D array of n2+1 increasing values that specify the pixel edges
            along the x2 direction.
        """
        self.x1_edges = np.asarray(x1_edges)
        self.x2_edges = np.asarray(x2_edges)
        self.shape = len(self.x2_edges) - 1, len(self.x1_edges) - 1
        # Check for increasing edge values.
        if np.any(np.diff(self.x1_edges) <= 0) or np.any(np.diff(self.x2_edges) <= 0):
            raise ValueError('Pixel edges are not strictly increasing.')
        # Calculate pixel areas.
        self.areas = np.diff(self.x2_edges).reshape(-1, 1) * np.diff(self.x1_edges)

    def gauss(self, mu1, mu2, sigma1, sigma2, rho, moments=False):
        """Calculate a single normalized Gaussian integrated over pixels.

        Parameters
        ----------
        mu1 : float
            Mean along x1
        m2 : float
            Mean along x2
        sigma1 : float
            Standard deviation along x1. Must be > 0.
        sigma2 : float
            Standard deviation along x2. Must be > 0.
        rho : float
            Correlation coefficient. Must be in the range (-1, +1).
        moments : bool
            Calculate the first-order moments <x1-mu1>, <x2-mu2> and
            second order moments <(x1-mu1)**2>, <(x1-mu1)*(x2-mu2)>,
            <(x2-mu2)**2> when True. Otherwise, only the area
            (zeroth-order moment) is calculated and returned.

        Returns
        -------
        array
            Array of shape (n2, n1) if moments is False, or (6, n2, n1)
            if moments is True.
        """
        if sigma1 <= 0:
            raise ValueError('sigma1 must be > 0.')
        if sigma2 <= 0:
            raise ValueError('sigma2 must be > 0.')
        if np.abs(rho) >= 1:
            raise ValueError('rho must be in (-1, +1).')
        n2, n1 = self.shape
        if moments:
            result = np.zeros((6, n2, n1))
        else:
            result = np.zeros((1, n2, n1))

        rho2 = rho ** 2
        c0 = 1 - rho2
        c1 = 2 / (np.pi * c0 ** 1.5)
        c2 = np.sqrt(2. / np.pi)
        denom = np.sqrt(2 * c0)

        # Normalize edge coordinate arrays.
        s1_edges = (self.x1_edges - mu1) / sigma1
        s2_edges = (self.x2_edges - mu2) / sigma2

        # Calculate midpoints and bin sizes, ready for broadcasting
        # as axis=0 in 2D array expressions.
        s1 = 0.5 * (s1_edges[1:] + s1_edges[:-1])[:, np.newaxis]
        ds1 = 0.5 * (s1_edges[1:] - s1_edges[:-1])[:, np.newaxis]
        s2 = 0.5 * (s2_edges[1:] + s2_edges[:-1])[:, np.newaxis]
        ds2 = 0.5 * (s2_edges[1:] - s2_edges[:-1])[:, np.newaxis]

        for transpose in (True, False):

            # Integrate zeroth-order moment using a series expansion in s1.
            norm = np.exp(-0.5 * s1 ** 2) * ds1 / 12.
            arg = (s2_edges - rho * s1) / denom
            exp_arg = np.exp(-arg ** 2)
            s1sq = s1 ** 2
            ds1sq = ds1 ** 2
            s2_rho = s2_edges * rho
            term1 = -c1 * rho * ds1sq * np.diff(
                exp_arg * (s2_rho - s1 * (2 - rho2)), axis=1)
            erf_diff = np.diff(scipy.special.erf(arg), axis=1)
            term2 = c2 * (6 + ds1sq * (s1sq - 1)) * erf_diff
            moment_0 = norm * (term1 + term2)

            # Optionally calculate higher-order moments.
            if moments:

                s1sqsq = s1sq ** 2

                # Calculate (x1-mu1) moment over each pixel.
                term1 = -c1 * rho * ds1sq * np.diff(
                    exp_arg * (2 * c0 + s1 *
                               (s2_rho - s1 * (2 - rho2))))
                term2 = c2 * s1 * (6 + ds1sq * (s1sq - 3)) * erf_diff
                moment_s1 = norm * (term1 + term2)

                # Calculate (x1-mu1)**2 moment over each pixel.
                term1 = -c1 * rho * ds1sq * s1 * np.diff(
                    exp_arg * (4 + s1 * s2_rho - 4 * rho2 -
                               s1sq * (2 - rho2)))
                term2 = c2 * (6 * s1sq + ds1sq *
                              (2 - 5 * s1sq + s1sqsq)) * erf_diff
                moment_s1s1 = norm * (term1 + term2)

                # Calculate (x1-mu1)(x2-mu2) moment over each pixel.
                term1 = -c1 * np.diff(
                    exp_arg * ds1sq * rho * (
                        s1 * s2_rho * (s2_edges - s1 * rho) +
                        c0 * (2 * (1 - s1sq) * s2_edges + s1 * rho)) +
                    (exp_arg - 1) * s1 * (6 + ds1sq * (s1sq - 3)) * c0 ** 2)
                term2 = c2 * rho * erf_diff * (
                    6 * s1sq + ds1sq * (2 - 5 * s1sq + s1sqsq))
                moment_s1s2 = norm * (term1 + term2)

            if transpose:
                result[0] = 0.5 * moment_0.T
                if moments:
                    result[1] = sigma1 * moment_s1.T
                    result[3] = sigma1 ** 2 * moment_s1s1.T
                    result[4] = 0.5 * sigma1 * sigma2 * moment_s1s2.T
                # Swap coordinates to calculate series expansion in s2.
                s1 = s2
                ds1 = ds2
                s2_edges = s1_edges
                sigma1, sigma2 = sigma2, sigma1
            else:
                result[0] += 0.5 * moment_0
                if moments:
                    result[2] = sigma1 * moment_s1
                    result[5] = sigma1 ** 2 * moment_s1s1
                    result[4] += 0.5 * sigma1 * sigma2 * moment_s1s2

        return result if moments else result[0]

    @staticmethod
    def get_Cinv(sigma1, sigma2, rho):
        """Calculate the inverse covariance matrix from model parameters.
        """
        S12 = sigma1 * sigma2
        C12 = S12 * rho
        detC = S12 ** 2 - C12 ** 2
        return np.array(((sigma2 ** 2, -C12), (-C12, sigma1 ** 2))) / detC

    @staticmethod
    def transform(pin, forward=True):
        """Transform the model parameters to/from internal unbounded parameters.
        """
        pout = pin.copy()
        deriv = np.ones_like(pout)
        k0 = len(pin) % 6
        (ufunc1, ufunc2) = (np.exp, np.tanh) if forward else (np.log, np.arctanh)
        if forward:
            for k in (0, 3, 4):
                pout[k0 + k::6] = np.exp(pin[k0 + k::6])
                deriv[k0 + k::6] = pout[k0 + k::6]
            pout[k0 + 5::6] = np.tanh(pin[k0 + 5::6])
            deriv[k0 + 5::6] = 2 / (np.cosh(2 * pin[k0 + 5::6]) + 1)
        else:
            for k in (0, 3, 4):
                pout[k0 + k::6] = np.log(pin[k0 + k::6])
                deriv[k0 + k::6] = 1 / pin[k0 + k::6]
            pout[k0 + 5::6] = np.arctanh(pin[k0 + 5::6])
            deriv[k0 + 5::6] = 1 / (1 - pin[k0 + 5::6] ** 2)
        return pout, deriv

    def predict(self, params, compute_partials=False):
        """Calculate a predicted model and (optionally) its partial derivatives.

        The number of Gaussians and the presence of a floating background are
        inferred from the input number of parameters, which are assumed to be:

        {BG} NORM[0] MU1[0] MU2[0] SIGMA1[0] SIGMA2[0] RHO[0] ...

        Parameters
        ----------
        params : numpy array
            Array of parameters to use.
        compute_partials : bool
            Compute partial derivative images with respect to each input parameter
            when True.

        Returns
        -------
        array or tuple
            Returns the predicted image D or a tuple (D, P) where P is an array of
            partial derivative images with respect to each of the input parameters.
        """
        nparams = len(params)
        if compute_partials:
            partials = np.empty((nparams,) + self.shape)
        ngauss = nparams // 6
        if nparams % 6 == 1:
            fitbg = True
            bgdensity, gauss_params = params[0], params[1:]
            result = bgdensity * self.areas
            if compute_partials:
                partials[0] = self.areas
                pnext = 1
        elif nparams % 6 == 0:
            fitbg = False
            gauss_params = params
            result = np.zeros(self.shape)
            pnext = 0
        else:
            raise ValueError('Invalid number of parameters.')
        norm, mu1, mu2, sigma1, sigma2, rho = gauss_params.reshape(ngauss, -1).T
        for k in range(ngauss):
            model = self.gauss(mu1[k], mu2[k], sigma1[k], sigma2[k], rho[k], moments=compute_partials)
            if compute_partials:
                Cinv = self.get_Cinv(sigma1[k], sigma2[k], rho[k])
                model, moments = model[0], model[1:]
                partials[pnext + 0] = model  # partial wrt norm[k]
                U = np.tensordot(Cinv, moments[0:2], axes=1)
                partials[pnext + 1] = norm[k] * U[0] # partial wrt mu1[k]
                partials[pnext + 2] = norm[k] * U[1] # partial wrt mu2[k]
                Q = np.stack((moments[2:4], moments[3:5]))
                M = np.einsum('ij,jkab,kl->ilab', Cinv, Q, Cinv) - np.einsum('ij,ab->ijab', Cinv, model)
                dC11 = norm[k] * M[0, 0] / 2 # partial wrt Cov11[k]
                dC22 = norm[k] * M[1, 1] / 2 # partial wrt Cov22[k]
                dC12 = norm[k] * M[0, 1]     # partial wrt Cov12[k]
                # Transform partials from Covij to sigma1, sigma2, rho.
                partials[pnext + 3] = 2 * sigma1[k] * dC11 + rho[k] * sigma2[k] * dC12 # partial wrt sigma1[k]
                partials[pnext + 4] = 2 * sigma2[k] * dC22 + rho[k] * sigma1[k] * dC12 # partial wrt sigma2[k]
                partials[pnext + 5] = sigma1[k] * sigma2[k] * dC12                     # partial wrt rho[k]
                pnext += 6
            result += norm[k] * model
        return (result, partials) if compute_partials else result

    def nll(self, params, data, ivar, compute_partials=False, transformed=False):
        """Calculate the negative-log-likelihood of the specified observation.

        Parameters
        ----------
        params : array
            1D array of parameters to use for the model prediction.
        data : array
            2D array of observed data values.
        ivar : array
            2D array of estimated inverse variances for each observed data value.
            Must have the same dimensions as data.
        compute_partials : bool
            Calculate and return partial derivatives of the result wrt to each
            input parameter when True.
        transformed : bool
            Apply non-linear transformations to compute bounded model parameter
            values from the unbounded input values. The exp transform is used
            for model parameters that must be > 0 (NORM, SIGMA1, SIGMA2) and
            the tanh transform is used for -1 < RHO < +1.  When compute_partials
            is True, the returned derivatives will be respect to the unbounded
            input parameters.

        Returns
        -------
        array or tuple
            Returns the calculated negative-log-likelihood nll or a tuple
            (nll, nll_partials) when compute_partials is True.
        """
        params = np.asarray(params)
        if transformed:
            # Transform from internal unbound params to model params and compute the
            # corresponding transform derivatives.
            params, derivs = self.transform(params)
        if compute_partials:
            predicted, partials = self.predict(params, compute_partials=True)
            nll_partials = 2 * np.sum(ivar * (predicted - data) * partials, axis=(1, 2))
            if transformed:
                # Transform the partial derivatives to be wrt the internal params.
                nll_partials *= derivs
        else:
            predicted = self.predict(params, compute_partials=False)
        nll = np.sum(ivar * (predicted - data) ** 2)
        return (nll, nll_partials) if compute_partials else nll
