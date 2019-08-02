import json
import collections
import functools

import numpy as np
import scipy.special

import desietcimg.util


def fiber_profile(x, y, r0, blur=0.1):
    """Radial profile of a blurred disk.

    This implementation approximates a 2D Gaussian blur using a 1D erf,
    so is only exact in the limit of zero blur (because the 2D
    Jacobian requires less blur to r > r0 than r < r0 to preserve area).
    This approximation means that we are assuming a slightly assymetric
    blur.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 + 0.5 * scipy.special.erf((r0 - r) / (np.sqrt(2) * blur))


class SkyCameraAnalysis(object):
    """ Initialize the Sky Camera image analysis.
    
    Parameters
    ----------
    nx : int
        Number of columns in a full frame image with no binning.
    ny : int
        Number of rows in a full frame image with no binning.
    binning : int
        Binning of the images to process. Must be in (1, 2, 3).
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
    def __init__(self, nx=3072, ny=2047, binning=1,
                 fiberdiam_um=107., pixelsize_um=9., blur_um=5.,
                 margin=2.0, search_pix=14, search_steps=29):
        self.ny = ny
        self.nx = nx
        self.binning = binning
        if binning not in (1, 2, 3):
            raise ValueError('Expected binning in (1, 2, 3).')
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
            fiber_profile, r0=0.5 * self.fiberdiam / binning, blur=self.blur / binning)
        for j, dy in enumerate(self.dxy):
            for i, dx in enumerate(self.dxy):
                self.T[j, i] = desietcimg.util.make_template(
                    ssize, profile, dx=dx, dy=dy,
                    oversampling=10 * binning, normalized=True)
        # Calculate the fiber area in binned pixels, for normalizing SNR calculations.
        self.fiber_area = np.pi * (0.5 * self.fiberdiam / binning) ** 2
        # We do not have any fiber location info yet.
        self.fibers = None

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
        self.mask = np.ones((self.ny // self.binning, self.nx // self.binning), bool)
        for (x, y) in self.fibers.values():
            x //= self.binning
            y //= self.binning
            xlo, ylo = x  - self.rsize, y - self.rsize
            xhi, yhi = x + self.rsize + 1, y + self.rsize + 1
            self.mask[ylo:yhi, xlo:xhi] = False
        if np.count_nonzero(~self.mask) != len(self.fibers) * (2 * self.rsize + 1) ** 2:
            raise ValueError('Fiber location constraints are violated.')
        
    def get_fiber_fluxes(self, data):
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

        Returns
        -------
        dict
            A dictionary of per-fiber results using the fiber labels as keys.
            Each entry provides (xfit, yfit, bgmean, fiber_flux, snr, stamp) where
            (xfit, yfit) are in units of unbinned pixels, (bgmean, fiber_flux)
            are in units of ADUs, SNR is as described above, and stamp is a copy
            of the postage stamp used for each fiber.
        """
        if self.fibers is None:
            raise RuntimeError('No fiber locations specified yet.')
        data = self.validate(data)
        # Estimate total noise over a fiber.
        noise_pixels, _, _ = scipy.stats.sigmaclip(data[self.mask])
        fiber_noise = np.sqrt(self.fiber_area * np.var(noise_pixels))
        # Assemble all fiber stamps into a single data array D.
        ssize = 2 * self.rsize + 1
        nfibers = len(self.fibers)
        D = np.empty((ssize ** 2, nfibers))
        stamps = []
        for k, (label, (x, y)) in enumerate(self.fibers.items()):
            # Extract the stamp centered on (x, y)
            ix = x // self.binning
            iy = y // self.binning
            stamp = data[iy - self.rsize:iy + self.rsize + 1,
                         ix - self.rsize:ix + self.rsize + 1].copy()
            D[:, k] = stamp.reshape(-1)
            stamps.append(stamp)
        # Loop over centroid hypotheses.
        nsteps = len(self.T)
        params = np.empty((nsteps, nsteps, 2, nfibers))
        scores = np.empty((nsteps, nsteps, nfibers))
        for j in range(nsteps):
            for i in range(nsteps):
                A = np.stack((np.ones(stamp.size), self.T[j, i].reshape(-1)), axis=1)
                # Solve the linear least-squares problem to find the mean noise and fiber flux
                # for this centroid hypothesis, simultaneously for all fibers.
                params[j, i], scores[j, i], _, _ = scipy.linalg.lstsq(A, D)
        # Find the maximum likelihood centroid for each stamp.
        idx = np.argmin(scores.reshape(nsteps ** 2, nfibers), axis=0)
        jbest, ibest = np.unravel_index(idx, (nsteps, nsteps))
        # Build the results.
        results = collections.OrderedDict()
        for k, (label, (x, y)) in enumerate(self.fibers.items()):
            # Convert grid indices to unbinned pixel coordinates.
            xfit = x + self.dxy[ibest[k]] * self.binning
            yfit = y + self.dxy[jbest[k]] * self.binning
            bgmean, fiber_flux = params[jbest[k], ibest[k], :, k]
            snr = fiber_flux / fiber_noise
            results[label] = (xfit, yfit, bgmean, fiber_flux, snr, stamps[k])
        # Save the results for plot_sky_camera.
        self.results = results
        return results


def init_signals(fibers, max_signal=1000., attenuation=0.95):
    signals = collections.OrderedDict()
    signal = max_signal
    for label in fibers:
        signals[label] = signal
        signal *= attenuation
    return signals


def add_fiber_signals(bg, signals, SCA, invgain=1.6, centroid_dxy=5, rng=None):
    """Add simulated fiber signals to a background image.
    """
    SCA.validate(bg)
    rng = rng or np.random.RandomState()
    # Construct pixel bin edges.
    ny, nx = bg.shape
    bins = (np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5)
    # Loop over fibers.
    truth = collections.OrderedDict()
    for label, (xfiber, yfiber) in SCA.fibers.items():
        # Add some random jitter.
        w = 0.5 * centroid_dxy
        xfiber += rng.uniform(-w, +w)
        yfiber += rng.uniform(-w, +w)
        # Convert to binned pixel coordinates.
        xfiber /= SCA.binning
        yfiber /= SCA.binning
        # Generate the number of detected electrons to use.
        nelec_mean = signals[label] * invgain
        nelec_det = rng.poisson(lam=nelec_mean)
        # Save true detected signal in ADU for this fiber.
        truth[label] = nelec_det / invgain
        # Generate electron positions w/o blur.
        r = 0.5 * SCA.fiberdiam / SCA.binning * np.sqrt(rng.uniform(size=nelec_det))
        phi = 2 * np.pi * rng.uniform(size=nelec_det)
        x0 = xfiber + r * np.cos(phi)
        y0 = yfiber + r * np.sin(phi)
        # Apply blur.
        x = rng.normal(loc=x0, scale=SCA.blur / SCA.binning)
        y = rng.normal(loc=y0, scale=SCA.blur / SCA.binning)
        # Bin in pixels.
        pixels, _, _ = np.histogram2d(x, y, bins=bins)
        # Convert to ADUs and add to the final image.
        bg += pixels.T / invgain
    return bg, truth
