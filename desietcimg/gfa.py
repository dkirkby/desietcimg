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
                    readnoise=None, gain=None, master_zero=None, mask=None, tempfit=None,
                    overwrite=True):
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
            hdus.write(mask[gfa].astype(np.uint8), extname='MASK{}'.format(gfanum))
    logging.info('Saved GFA calib data to {0}.'.format(name))


def load_calib_data(name='GFA_calib.fits'):
    data = {}
    master_zero = {}
    mask = {}
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
            for key in 'TREF', 'IREF', 'TCOEF':
                data[gfa][key] = hdr[key]
            master_zero[gfa] = hdus['ZERO{0}'.format(gfanum)].read().copy()
            mask[gfa] = hdus['MASK{0}'.format(gfanum)].read().astype(np.bool)
    logging.info('Loaded GFA calib data from {0}.'.format(name))
    return data, master_zero, mask


class GFACamera(object):

    gfa_names = [
        'GUIDE0', 'FOCUS1', 'GUIDE2', 'GUIDE3', 'FOCUS4',
        'GUIDE5', 'FOCUS6', 'GUIDE7', 'GUIDE8', 'FOCUS9']
    amp_names = ['E', 'F', 'G', 'H']
    lab_data = None
    calib_data = None
    master_zero = None
    mask = None

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
            GFACamera.calib_data, GFACamera.master_zero, GFACamera.mask = load_calib_data(calib_name)

    def setraw(self, raw, name=None, subtract_master_zero=True, apply_gain=True):
        """Initialize using the raw GFA data provided, which can either be a single or multiple exposures.

        After calling this method the following attributes are set:

            nexp : int
                Number of exposures loaded, which will be one if raw is a 2D array.
            bias : dict of arrays
                Bias values in ADU estimated from the overscan in each exposure, indexed by the amplifier name.
            amps : dict of view
                Raw array views indexed by amplifier name, including pre and post overscan regions, in row
                and column readout order.
            data : 3D array of float32
                Bias subtracted pixel values in ADU of shape (nexp, 2 * nampy, 2 * nampx) with
                pre and post overscan regions removed from the raw data.

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
        # Assemble the bias-subtracted data with overscan removed.
        self.data[:, :self.nampy, :self.nampx] = raw[:, :self.nampy, self.nscan:self.nampx + self.nscan] - self.bias['E'].reshape(-1, 1, 1)
        self.data[:, :self.nampy, self.nampx:] = raw[:, :self.nampy, self.nxby2 + self.nscan:-self.nscan] - self.bias['F'].reshape(-1, 1, 1)
        self.data[:, self.nampy:, :self.nampx] = raw[:, self.nampy:, self.nscan:self.nampx + self.nscan] - self.bias['H'].reshape(-1, 1, 1)
        self.data[:, self.nampy:, self.nampx:] = raw[:, self.nampy:, self.nxby2 + self.nscan:-self.nscan] - self.bias['G'].reshape(-1, 1, 1)
        # Subtract the master zero if requested.
        if subtract_master_zero:
            self.data -= GFACamera.master_zero[name]
        # Apply the gain correction if requested.
        if apply_gain:
            for amp in self.amp_names:
                self.data[self.quad[amp][1:]] *= GFACamera.calib_data[name][amp]['GAIN']
            self.unit = 'elec'
        else:
            self.unit = 'ADU'
