import json
import collections
import functools

import numpy as np

import scipy.special
import scipy.signal
import scipy.stats

import desietcimg.util


class SkyCameraAnalysis(object):
    """ Initialize the Sky Camera image analysis.

    Parameters
    ----------
    calib : desietcimg.calib.CalibrationAnalysis
        Calibration results for camera data to analyze.
    fiberdiam_um : float
        Fiber diameter in microns.
    pixelsize_um : float
        Phyiscal (unbinned) pixel size in microns.
    blur_pix : float
        Focus blur of fiber tips in microns.
    margin : float
        Postage stamp cutouts are scaled to be larger than the fiber by this factor.
    search_pix : float
        Size of centroid search region to scan in units of unbinned pixels.
    search_steps : int
        Number of search grid points to use covering the search region.
    """
    def __init__(self, calib, fiberdiam_um=107., pixelsize_um=9., blur_um=5.,
                 margin=2.0, search_pix=10, search_steps=21):
        ny, nx = calib.shape
        if nx == 3072:
            binning = 1
        elif nx == 1536:
            binning = 2
        elif nx == 1024:
            binning = 3
        else:
            raise RuntimeError('Unable to infer binning from nx={0}.'.format(nx))
        self.ny = ny * binning
        self.nx = nx * binning
        self.binning = binning
        self.invgain = calib.flatinvgain
        self.pixmask = (calib.pixmask != 0)
        self.darkmu = calib.darkmu.copy()
        self.darkvar = calib.darkvar.copy()
        # Convert fiber diameter and blur to (unbinned) pixels.
        self.fiberdiam = fiberdiam_um / pixelsize_um
        self.blur = blur_um / pixelsize_um
        # Calculate the stamp size to use as ssize = 2 * rsize + 1 (always odd).
        self.rsize = int(np.ceil((margin * 0.5 * self.fiberdiam + 3 * self.blur) / binning))
        ssize = 2 * self.rsize + 1
        # Build templates in binned pixels for different sub-pixel offsets.
        self.dxy = np.linspace(-0.5 * search_pix / binning,
                               +0.5 * search_pix / binning, search_steps)
        self.T = np.empty((search_steps, search_steps, ssize, ssize))
        profile = functools.partial(
            desietcimg.util.fiber_profile, r0=0.5 * self.fiberdiam / binning, blur=self.blur / binning)
        for j, dy in enumerate(self.dxy):
            for i, dx in enumerate(self.dxy):
                self.T[j, i] = desietcimg.util.make_template(
                    ssize, profile, dx=dx, dy=dy,
                    oversampling=10 * binning, normalized=True)
        # Calculate the fiber area in binned pixels, for normalizing SNR calculations.
        self.fiber_area = np.pi * (0.5 * self.fiberdiam / binning) ** 2
        # We do not have any fiber location info yet.
        self.fibers = None
        self.results = None

    def validate(self, data):
        """
        """
        data = np.asarray(data)
        ny, nx = data.shape
        if ny != self.ny // self.binning or nx != self.nx // self.binning:
            raise ValueError('Input data has unexpected shape: {0}.'.format(data.shape))
        return data

    def find_fiber_locations(self, data, nfibers=18, savename=None):
        """Find fiber locations as the brightest points with the expected shape.

        The algorithm convolves the data with a filter matched to the fiber size
        and will always return the requested number of locations subject to the
        constraints that no two fibers are within a stamp size of each other
        and no fiber is within rsize of the image edge.

        To get reliable results, all fibers should have sufficiently bright images.
        The located centroids should then be accurate to ~1 binned pixel.

        Parameters
        ----------
        data : array
            2D array of image data with shape (ny, nx).
        nfibers : int
            The number of fibers to locate.
        savename : string or None
            Save the results to the specified filename in the json format expected
            by :meth:`load_fiber_locations` when not None. Note that labels
            will be assigned in order of decreasing brightness and will not
            be matched to petal locations.

        Returns
        -------
        tuple
            Tuple (xfiber, yfiber) of nfiber centroids in unbinned pixel coordinates.
        """
        data = self.validate(data)
        ny, nx = data.shape
        # Use a template for a fiber ~centered in the stamp.
        nby2 = len(self.T) // 2
        T0 = self.T[nby2, nby2]
        # Convolve the image with this template.
        filtered = scipy.signal.convolve(data, T0, mode='valid')
        nyf, nxf = filtered.shape
        assert ny == nyf + 2 * self.rsize and nx == nxf + 2 * self.rsize
        clear_value = np.min(filtered)
        # Locate fibers in decreasing order of template correlation.
        xfiber, yfiber = [], []
        while len(xfiber) < nfibers:
            # Find the largest (filtered) pixel value.
            iy, ix = np.unravel_index(np.argmax(filtered), (nyf, nxf))
            xfiber.append(self.binning * (ix + self.rsize))
            yfiber.append(self.binning * (iy + self.rsize))
            # Clear pixels in the stamp centered on this pixel.
            xlo, ylo = max(0, ix - self.rsize), max(0, iy - self.rsize)
            xhi, yhi = min(nxf, ix + self.rsize + 1), min(nyf, iy + self.rsize + 1)
            filtered[ylo:yhi, xlo:xhi] = clear_value
        if savename is not None:
            labels = ['SKY{0}'.format(k) for k in range(nfibers)]
            fibers = collections.OrderedDict()
            for k, label in enumerate(labels):
                fibers[label] = int(xfiber[k]), int(yfiber[k])
            with open(savename, 'w') as f:
                json.dump(fibers, f, indent=4)
        return xfiber, yfiber

    def load_fiber_locations(self, name='fibers.json'):
        """Load fiber locations from a json file.

        All fiber locations must be inset at least rsize from the image
        borders and at least 2 * rsize + 1 pixels away from other locations.

        This method must be called before :meth:`get_fiber_fluxes`.

        Parameters
        ----------
        name : str
            Name of the json file to load.
        """
        path = desietcimg.util.get_data(name, must_exist=True)
        with open(path, 'r') as f:
            self.fibers = json.load(f, object_pairs_hook=collections.OrderedDict)
        # Build a mask to select background pixels for noise measurement.
        self.bgmask = np.ones((self.ny // self.binning, self.nx // self.binning), bool)
        self.fiberslice = []
        for (x, y) in self.fibers.values():
            x //= self.binning
            y //= self.binning
            xlo, ylo = x  - self.rsize, y - self.rsize
            xhi, yhi = x + self.rsize + 1, y + self.rsize + 1
            # Remember the slice for this stamp.
            fslice = (slice(ylo, yhi), slice(xlo, xhi))
            self.fiberslice.append(fslice)
            self.bgmask[fslice] = False
        if np.count_nonzero(~self.bgmask) != len(self.fibers) * (2 * self.rsize + 1) ** 2:
            raise ValueError('Fiber location constraints are violated.')
        # Removed masked pixels from  the background.
        self.bgmask[self.pixmask] = False

    def get_fiber_fluxes(self, data, exptime, maxrate=1000., refit=False, chisq_cut=1e8):
        """Estimate fiber fluxes and SNR values.

        Scans the search window for each fiber to identify the maximum
        likelihood fiber centroid position and calcualtes the corresponding
        background level and integrated fiber flux (using linear least squares).
        
        The signal-to-noise ratios are calculated assuming a noise of
        pixel_rms * sqrt(fiber_area) where pixel_rms is the sigma-clipped
        standard deviation of pixel values outside each fiber stamp.

        Uses the fiber locations from the most recent call to
        :meth:`load_fiber_locations`.

        Parameters
        ----------
        data : array
            2D array of image data with shape (ny, nx).
        exptime : float
            Exposure time in seconds.
        maxrate : float
            Maximum expected rate for any fiber in elec/sec.
        refit : bool
            When True, refit when a stamp appears to be contaminated by
            a cosmic ray (according to chisq_cut).
        chisq_cut : float
            Pixels whose contribution to the best-fit chisq exceed this
            cut are masked and the fit is repeated. The purpose of this
            is to be somewhat robust in the presence of cosmics.

        Returns
        -------
        dict
            A dictionary of per-fiber results using the fiber labels as keys.
            Each entry provides (xfit, yfit, bgmean, fiber_flux, snr, stamp) where
            (xfit, yfit) are in units of unbinned pixels, bgmean is in units of ADUs,
            fiber_flux is in elec/sec, SNR is as described above, and stamp is a copy
            of the postage stamp used for each fiber.
        """
        if self.fibers is None:
            raise RuntimeError('No fiber locations specified yet.')
        data = self.validate(data)
        # Estimate integrated noise over a fiber in ADU.
        noise_pixels, _, _ = scipy.stats.sigmaclip(data[self.bgmask])
        noise_var = np.var(noise_pixels)
        fiber_noise = np.sqrt(self.fiber_area * noise_var)
        # Assemble all fiber stamps into a single data array D.
        ssize = 2 * self.rsize + 1
        nfibers = len(self.fibers)
        results = collections.OrderedDict()
        # Convert maxrate to a maximum integrated signal in ADU.
        fmax = maxrate * exptime / self.invgain
        for k, (label, (x, y)) in enumerate(self.fibers.items()):
            # Extract the stamp centered on (x, y)
            fslice = self.fiberslice[k]
            raw = data[fslice].copy()
            # Subtract the per-pixel average dark.
            stamp = raw.astype(float)
            stamp -= self.darkmu[fslice]
            # Look up the variances for this stamp.
            stampvar = self.darkvar[fslice]
            # Mask bad pixels.
            mask = self.pixmask[fslice]
            # Measure the flux and background level in this fiber.
            (dxfit, dyfit, bgfit, ffit, ivar, model) = self.measure_one_fiber(stamp, stampvar, mask, fmax)
            if refit:
                # Calculate per-pixel contributions to the chisquare.
                chisq = ivar * (stamp - model) ** 2
                bad_chisq = chisq > chisq_cut
                if np.any(bad_chisq):
                    # Mask the pixels above the cut.
                    mask[bad_chisq] = True
                    # Refit if there are enough pixel remaining.
                    if np.count_nonzero(mask) < 0.25 * ssize ** 2:
                        (dxfit, dyfit, bgfit, ffit, ivar, model) = self.measure_one_fiber(stamp, stampvar, mask, fmax)
            # Convert offsets in binned pixels to absolute unbinned pixel coordinates.
            xfit = x + dxfit * self.binning
            yfit = y + dyfit * self.binning
            # Estimate the SNR.
            snr = ffit / fiber_noise
            # Convert integrated flux in ADU to a rate in elec/sec.
            fiber_flux = ffit * self.invgain / exptime
            # Save the results of this fiber.
            stamp[mask] = 0
            results[label] = (xfit, yfit, bgfit, fiber_flux, snr, stamp, ivar, model, raw)
        # Save the results for plot_sky_camera.
        self.results = results
        return results

    def measure_one_fiber(self, stamp, stampvar, mask, fmax):
        """Measure the integrated flux in a single fiber.

        Parameters
        ----------
        stamp : array
            2D array of pixel data for a stamp with dimensions (ssize, ssize).
        stampvar : array
            2D array of corresponding background inverse variances.
        mask : array
            2D array of boolean mask values.
        exptime : float
            Exposure time in seconds. Used to normalize the measurement as a rate.
        maxrate : float
            Maximum expected rate for any fiber in elec/sec.
        """
        # Make a rough estimate of the fiber flux.
        f0 = max(0., stamp.sum())
        # Make sure this corresponds to a reasonable rate.
        f0 = min(f0, fmax)
        # Loop over centroid hypotheses.
        nsteps = len(self.T)
        params = np.empty((nsteps, nsteps, 2))
        scores = np.empty((nsteps, nsteps))
        for j in range(nsteps):
            for i in range(nsteps):
                # Look up the template model for this centroid.
                M = self.T[j, i]
                # Estimate the inverse variance for this centroid hypothesis assuming the
                # (fixed) rough estimate of the fiber flux.
                var = stampvar + f0 * M / self.invgain
                ivar = np.divide(1, var, out=np.zeros_like(var), where=var>0)
                ivar[mask] = 0
                # Construct the weighted linear least squares problem.
                w = np.sqrt(ivar)
                A = np.stack((np.ones(stamp.size), M.reshape(-1)), axis=1) * w.reshape(-1, 1)
                y = (stamp * w).reshape(-1)
                # Solve the weighted linear least-squares problem to find the mean background
                # level and fiber flux for this centroid hypothesis.
                params[j, i, :], scores[j, i], _, _ = scipy.linalg.lstsq(A, y)
        # Find the maximum likelihood centroid.
        idx = np.argmin(scores.reshape(nsteps ** 2))
        jbest, ibest = np.unravel_index(idx, (nsteps, nsteps))
        # Convert grid indices to binned pixel offsets relative to the stamp center.
        dxfit, dyfit = self.dxy[ibest], self.dxy[jbest]
        # Lookup the best-fit background mean in ADU and integrated fiber flux in ADU.
        bgfit, ffit = params[jbest, ibest]
        # Calculate the ivar for the best-fit centroid.
        M = self.T[jbest, ibest]
        var = stampvar + ffit * M / self.invgain
        ivar = np.divide(1, var, out=np.zeros_like(var), where=var>0)
        ivar[mask] = 0
        # Calculate the best-fit model.
        model = bgfit + ffit * M
        return (dxfit, dyfit, bgfit, ffit, ivar, model)


def init_signals(fibers, rate1, rate2, logsteps=False):
    """Initialize signals that vary from rate1 to rate2.
    """
    signals = collections.OrderedDict()
    if logsteps:
        drate = (rate2 / rate1) ** (1 / (len(fibers) - 1))
    else:
        drate = (rate2 - rate1) / (len(fibers) - 1)
    for k, label in enumerate(fibers):
        if logsteps:
            signals[label] = rate1 * drate ** k
        else:
            signals[label] = rate1 + drate * k
    return signals


def add_fiber_signals(bg, signals, SCA, exptime, invgain, centroid_dxy=5, rng=None):
    """Add simulated fiber signals to a background image.
    """
    SCA.validate(bg)
    rng = rng or np.random.RandomState()
    # Define fiber radial profile.
    profile = functools.partial(
        desietcimg.util.fiber_profile, r0=0.5 * SCA.fiberdiam / SCA.binning, blur=SCA.blur / SCA.binning)
    # Initialize a postage stamp that covers each fiber.
    ssize = 2 * SCA.rsize + 1
    # Loop over fibers.
    data = bg.copy()
    w = 0.5 * centroid_dxy
    truth = collections.OrderedDict()
    for label, (xfiber, yfiber) in SCA.fibers.items():
        # Add some random jitter.
        xfiber += rng.uniform(-w, +w)
        yfiber += rng.uniform(-w, +w)
        # Convert to binned pixel coordinates.
        xfiber /= SCA.binning
        yfiber /= SCA.binning
        # Find the nearest binned pixel center.
        ix, iy = int(round(xfiber)), int(round(yfiber))
        # Calculate the mean fraction of photons in each postage stamp pixel.
        frac = desietcimg.util.make_template(
            ssize, profile, dx=xfiber - ix, dy=yfiber - iy,
            oversampling=10 * SCA.binning, normalized=True)
        # Generate a Poisson sample of detected electrons in each pixel.
        nelec_tot = signals[label] * exptime
        nelec = rng.poisson(lam=frac * nelec_tot)
        # Record the true number of detected electrons.
        truth[label] = nelec.sum() / exptime
        # Convert to ADU and add to the background image.
        data[iy - SCA.rsize:iy + SCA.rsize + 1,
           ix - SCA.rsize:ix + SCA.rsize + 1] += (nelec / invgain).astype(data.dtype)
    return data, truth
