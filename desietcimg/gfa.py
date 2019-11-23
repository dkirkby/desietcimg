"""Guide Focus Array (GFA) Utilities
"""
import logging

import numpy as np

import fitsio

import desietcimg.util


def load_lab_data(filename='GFA_lab_data.csv'):
    lab_data = {}
    path = desietcimg.util.get_data(filename, must_exist=True)
    csv_data = np.genfromtxt(
        path, delimiter=',', names=True,
        dtype=['U6', 'U11', 'i2', 'i2', 'U1', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'U100'])
    for gfa in np.unique(csv_data['GFA']):
        sel = np.where(csv_data['GFA'] == gfa)[0]
        assert len(sel) == 4
        first = csv_data[sel[0]]
        lab_data[gfa] = {
            'CCD': first['CCD'],
            'FILTID': first['FILTID'],
            'REF': first['REF'],
        }
        for amp, idx in zip('EFGH', sel):
            row = csv_data[idx]
            lab_data[gfa][amp] = {
                'RDNOISE': row['RDNOISE_e'],
                'FWELL': row['FWELL_Ke'],
                'GAIN': row['GAIN_eADU'],
            }
    logging.info('Loaded GFA lab data from {0}.'.format(path))
    return lab_data


def save_calib_data(name='GFA_calib.fits', comment='GFA in-situ calibration results',
                    readnoise=None, gain=None, master_zero=None, pixel_mask=None, tempfit=None,
                    master_dark=None, overwrite=True):
    """Any elements left blank will be copied from the current default calib data.
    """
    GFA = desietcimg.gfa.GFACamera()
    if master_zero is None:
        print('Using default master_zero')
        master_zero = GFA.master_zero
    if master_dark is None:
        print('Using default master_dark')
        master_dark = GFA.master_dark
    if pixel_mask is None:
        print('Using default pixel_mask')
        pixel_mask = GFA.pixel_mask
    _readnoise, _gain, _tempfit = {}, {}, {}
    for gfa in GFA.gfa_names:
        _readnoise[gfa] = {}
        _gain[gfa] = {}
        _tempfit[gfa] = {}
        for amp in GFA.amp_names:
            calib = GFA.calib_data[gfa][amp]
            _readnoise[gfa][amp] = calib['RDNOISE']
            _gain[gfa][amp] = calib['GAIN']
        calib = GFA.calib_data[gfa]
        for k in 'TREF', 'IREF', 'TCOEF', 'I0', 'C0':
            _tempfit[gfa][k] = calib[k]
    if readnoise is None:
        print('Using default readnoise')
        readnoise = _readnoise
    if gain is None:
        print('Using default gain')
        gain = _gain
    if tempfit is None:
        print('Using default tempfit')
        tempfit = _tempfit
    with fitsio.FITS(name, 'rw', clobber=overwrite) as hdus:
        # Write a primary HDU with only the comment.
        hdus.write(np.zeros((1,), dtype=np.float32), header=dict(COMMENT=comment))
        # Loop over GFAs.
        for gfanum, gfa in enumerate(desietcimg.gfa.GFACamera.gfa_names):
            hdr = {}
            for amp in desietcimg.gfa.GFACamera.amp_names:
                hdr['RDNOISE_{0}'.format(amp)] = readnoise[gfa][amp]
                hdr['GAIN_{0}'.format(amp)] = gain[gfa][amp]
            # Add dark current temperature fit results.
            for k, v in tempfit[gfa].items():
                hdr[k] = v
            # Write the per-GFA image arrays.
            hdus.write(master_zero[gfa], header=hdr, extname='ZERO{}'.format(gfanum))
            hdus.write(master_dark[gfa], extname='DARK{}'.format(gfanum))
            hdus.write(pixel_mask[gfa].astype(np.uint8), extname='MASK{}'.format(gfanum))
    print('Saved GFA calib data to {0}.'.format(name))


def load_calib_data(name='GFA_calib.fits'):
    data = {}
    master_zero = {}
    master_dark = {}
    pixel_mask = {}
    with fitsio.FITS(name) as hdus:
        # Loop over GFAs.
        for gfanum, gfa in enumerate(desietcimg.gfa.GFACamera.gfa_names):
            hdr = hdus['ZERO{0}'.format(gfanum)].read_header()
            data[gfa] = {}
            for amp in desietcimg.gfa.GFACamera.amp_names:
                data[gfa][amp] = {
                    'RDNOISE': hdr['RDNOISE_{0}'.format(amp)],
                    'GAIN': hdr['GAIN_{0}'.format(amp)],
                }
            for key in 'TREF', 'IREF', 'TCOEF', 'I0', 'C0':
                data[gfa][key] = hdr.get(key, -1)
            master_zero[gfa] = hdus['ZERO{0}'.format(gfanum)].read().copy()
            master_dark[gfa] = hdus['DARK{0}'.format(gfanum)].read().copy()
            pixel_mask[gfa] = hdus['MASK{0}'.format(gfanum)].read().astype(np.bool)
    logging.info('Loaded GFA calib data from {0}.'.format(name))
    return data, master_zero, master_dark, pixel_mask


class GFACamera(object):

    gfa_names = [
        'GUIDE0', 'FOCUS1', 'GUIDE2', 'GUIDE3', 'FOCUS4',
        'GUIDE5', 'FOCUS6', 'GUIDE7', 'GUIDE8', 'FOCUS9']
    amp_names = ['E', 'F', 'G', 'H']
    lab_data = None
    calib_data = None
    master_zero = None
    master_dark = None
    pixel_mask = None

    def __init__(self, nampy=516, nampx=1024, nscan=50, nrowtrim=4, maxdelta=50, calib_name='GFA_calib.fits'):

        self.nampy = nampy
        self.nampx = nampx
        self.nscan = nscan
        self.nxby2 = nampx + 2 * nscan
        self.nrowtrim = nrowtrim
        self.maxdelta = maxdelta
        self.data = None
        self.quad = {
            'E': (slice(None), slice(None, self.nampy), slice(None, self.nampx)), # bottom left
            'H': (slice(None), slice(self.nampy, None), slice(None, self.nampx)), # top left
            'F': (slice(None), slice(None, self.nampy), slice(self.nampx, None)), # bottom left
            'G': (slice(None), slice(self.nampy, None), slice(self.nampx, None)), # top left
        }
        # Load the class-level lab and calib data if necessary.
        if GFACamera.lab_data is None:
            GFACamera.lab_data = load_lab_data()
        if GFACamera.calib_data is None:
            (GFACamera.calib_data, GFACamera.master_zero,
             GFACamera.master_dark, GFACamera.pixel_mask) = load_calib_data(calib_name)
        # We have no exposures loaded yet.
        self.nexp = 0

    def setraw(self, raw, name=None, overscan_correction=True, subtract_master_zero=True, apply_gain=True):
        """Initialize using the raw GFA data provided, which can either be a single or multiple exposures.

        After calling this method the following attributes are set:

            nexp : int
                Number of exposures loaded, which will be one if raw is a 2D array.
            bias : dict of arrays
                Bias values in ADU estimated from the overscan in each exposure, indexed by the amplifier name.
            amps : dict of view
                Raw array views indexed by amplifier name, including pre and post overscan regions, in row
                and column readout order.
            unit : str
                Either 'elec' or 'ADU' depending on the value of apply_gain.
            data : 3D array of float32
                Bias subtracted pixel values in elec (or ADU if apply_gain is False) of shape
                (nexp, 2 * nampy, 2 * nampx) with pre and post overscan regions removed from the raw data.
            ivar : 3D array of float32
                Inverse variance estimated for each exposure in units matched to the data array.

        Parameters:
            raw : numpy array
                An array of raw data with shape (nexp, ny, nx) or (ny, nx). The raw input is not copied
                or modified.
            name : str or None
                Name of the camera that produced this raw data. Must be set to one of the values in gfa_names
                in order to lookup the correct master zero and dark images, and amplifier parameters, when
                these features are used.
            subtract_master_zero : bool
                Subtract the master zero image for this camera after applying overscan bias correction.
                Note that the overscan bias correction is always applied.
            apply_gain : bool
                Convert from ADU to electrons using the gain specified for this camera.
        """
        if raw.ndim not in (2, 3):
            raise ValueError('raw data must be 2D or 3D.')
        raw_shape = (2 * self.nampy, 2 * self.nampx + 4 * self.nscan)
        if raw.shape[-2:] != raw_shape:
            raise ValueError('raw data has dimensions {0} but expected {1}.'.format(raw.shape[-2:], raw_shape))
        if raw.ndim == 2:
            raw = raw.reshape((1,) + raw_shape)
        self.nexp, ny, nx = raw.shape
        if name not in self.gfa_names:
            logging.warning('Not a valid GFA name: {0}.'.format(name))
        self.name = name
        # Create views (with no data copied) for each amplifier with rows and column in readout order.
        self.amps = {
            'E': raw[:, :self.nampy, :self.nxby2], # bottom left (using convention that raw[0,0] is bottom left)
            'H': raw[:, -1:-(self.nampy + 1):-1, :self.nxby2], # top left
            'F': raw[:, :self.nampy, -1:-(self.nxby2+1):-1], # bottom right
            'G': raw[:, -1:-(self.nampy + 1):-1, -1:-(self.nxby2+1):-1], # top right
        }
        assert all((self.amps[ampname].base is raw for ampname in self.amp_names))
        # Calculate bias as mean overscan in each exposure, ignoring the first nrowtrim rows
        # (in readout order) and any values > maxdelta from the per-exposure median overscan.
        # Since we use a mean rather than median, subtracting this bias changes the dtype from
        # uint32 to float32 and means that digitization noise averages out over exposures.
        self.bias = {}
        for amp in self.amp_names:
            overscan = self.amps[amp][:, self.nrowtrim:, -self.nscan:]
            delta = overscan - np.median(overscan, axis=(1, 2), keepdims=True)
            bad = np.abs(delta) > self.maxdelta
            ngood = np.full(self.nexp, (self.nampy - self.nrowtrim) * self.nscan)
            if np.any(bad):
                nbad = np.count_nonzero(bad, axis=(1, 2))
                logging.warning('Ignoring {0} bad overscan pixels for {1}-{2}.'
                    .format(nbad.sum(), name, amp))
                overscan = np.copy(overscan)
                overscan[bad] = 0.
                ngood -= nbad
            self.bias[amp] = np.sum(overscan, axis=(1, 2)) / ngood
        # Only allocate new memory if necessary.
        if self.data is None or len(self.data) != self.nexp:
            self.data = np.empty((self.nexp, 2 * self.nampy, 2 * self.nampx), np.float32)
            self.ivar = np.empty((self.nexp, 2 * self.nampy, 2 * self.nampx), np.float32)
        # Assemble the real pixel data with the pre and post overscans removed.
        self.data[:, :self.nampy, :self.nampx] = raw[:, :self.nampy, self.nscan:self.nampx + self.nscan]
        self.data[:, :self.nampy, self.nampx:] = raw[:, :self.nampy, self.nxby2 + self.nscan:-self.nscan]
        self.data[:, self.nampy:, :self.nampx] = raw[:, self.nampy:, self.nscan:self.nampx + self.nscan]
        self.data[:, self.nampy:, self.nampx:] = raw[:, self.nampy:, self.nxby2 + self.nscan:-self.nscan]
        if overscan_correction:
            # Apply the overscan bias corrections.
            self.data[:, :self.nampy, :self.nampx] -= self.bias['E'].reshape(-1, 1, 1)
            self.data[:, :self.nampy, self.nampx:] -= self.bias['F'].reshape(-1, 1, 1)
            self.data[:, self.nampy:, :self.nampx] -= self.bias['H'].reshape(-1, 1, 1)
            self.data[:, self.nampy:, self.nampx:] -= self.bias['G'].reshape(-1, 1, 1)
        # Subtract the master zero if requested.
        if subtract_master_zero:
            self.data -= GFACamera.master_zero[name]
        # Apply the gain correction if requested.
        if apply_gain:
            calib = GFACamera.calib_data[name]
            for amp in self.amp_names:
                self.data[self.quad[amp]] *= calib[amp]['GAIN']
            # Use the calculated signal in elec as the estimate of Poisson variance.
            self.ivar[:] = self.data
            # Add the per-amplifier readnoise to the variance.
            for amp in self.amp_names:
                rdnoise_in_elec = calib[amp]['RDNOISE'] * calib[amp]['GAIN']
                self.ivar[self.quad[amp]] += rdnoise_in_elec ** 2
            # Convert var to ivar in-place, avoiding divide by zero.
            np.divide(1, self.ivar, out=self.ivar, where=self.ivar > 0)
            # Zero ivar for any masked pixels.
            self.ivar[:, self.pixel_mask[name]] = 0
            self.unit = 'elec'
        else:
            self.unit = 'ADU'

    def get_dark_current(self, ccdtemp=None, exptime=None, method='decorrelate', name=None, retval='image'):
        """Calculate the predicted dark current as a scaled master dark image.

        Parameters
        ----------
        ccdtemp : float or array or None
            The CCD temperature to subtract in degC, normally taken from the GCCDTEMP FITS
            header keyword.  If multiple exposures are loaded, can be an array or constant.
            The value None is only allowed whem method is 'decorrelate'.
        exptime : float or array or None
            The exposure time in seconds, normally taken from the EXPTIME FITS header
            keyword.  If multiple exposures are loaded, can be an array or constant.
            The value None is only allowed whem method is 'decorrelate'.
        method : 'linear' or 'exp' or 'decorrelate'
            When 'decorrelate', determine the effective integration time at 11C by setting
            the weighted correlation of the data with the master dark to zero.  This method
            does not require any input temperature or exposure time but does require that
            some raw data has already been loaded with :meth:`setraw`. Otherwise, use the
            fitted linear or exponential (Arrhenius) model to correct for temperature at the
            specified exposure time. These methods require that ``ccdtemp`` and ``exptime``
            values are provided, but do not require (or use) any previously loaded raw data.
        name : str or None
            Assume the specified camera. When None, use the name specified for the most
            recent call to :meth:`setraw`.
        retval : 'image' or 'frac'
            Returns the dark current images in electrons for each exposure as a 3D array
            for 'image', or the corresponding fractions of the master dark image when 'frac'.
            These fractions can be interpreted as the effective integration time in
            seconds for the dark current at TREF (nominally 11C).

        Returns
        -------
        array
            3D array of predicted dark current in electrons with shape (nexp, ny, nx).
        """
        if method == 'decorrelate':
            if self.nexp == 0 or self.unit != 'elec':
                raise RuntimeError('The decorrelate method needs raw data converted to electrons.')
        else:
            ccdtemp = np.atleast_1d(ccdtemp)
            exptime = np.atleast_1d(exptime)
        # Look up the temperature model coefficients for this camera.
        name = name or self.name
        if name not in self.gfa_names:
            raise RuntimeError('Cannot subtract dark current from unknown camera: "{0}".'.format(name))
        master = self.master_dark[name]
        calib = self.calib_data[self.name]
        # Calculate the predicted and reference average dark currents in elec/s.
        if method == 'linear':
            # The IREF parameter cancels in the ratio.
            TCOEF, TREF = calib['TCOEF'], calib['TREF']
            ratio = 1 + TCOEF * (ccdtemp - TREF)
            frac = exptime * ratio
        elif method == 'exp':
            # The I0 parameter cancels in the ratio.
            C0, TREF = calib['C0'], calib['TREF']
            ratio = np.exp(-C0 / (ccdtemp + 273.15)) / np.exp(-C0 / (TREF + 273.15))
            frac = exptime * ratio
        elif method == 'decorrelate':
            # Calculate the fraction of the template to subtract in order to
            # achieve zero weighted corelation with the template.
            T = (self.ivar *  master).reshape(self.nexp, -1)
            T /= np.sum(T ** 2, axis=1, keepdims=True)
            WD = (self.data * self.ivar).reshape(self.nexp, -1)
            frac = np.sum(WD * T, axis=1)
        else:
            raise ValueError('Invalid method "{0}".'.format(method))
        if retval == 'image':
            return master * frac.reshape(-1, 1, 1)
        elif retval == 'frac':
            return frac
        else:
            raise ValueError('Invalid retval "{0}".'.format(retval))
