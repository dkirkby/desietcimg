"""Guide Focus Array (GFA) Utilities
"""
import numpy as np

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
    return lab_data


class GFACamera(object):

    gfa_names = [
        'GUIDE0', 'FOCUS1', 'GUIDE2', 'GUIDE3', 'FOCUS4',
        'GUIDE5', 'FOCUS6', 'GUIDE7', 'GUIDE8', 'FOCUS9']
    amp_names = ['E', 'F', 'G', 'H']
    lab_data = None

    def __init__(self, nampy=516, nampx=1024, nscan=50, nrowtrim=4, maxdelta=50):

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
        # Load the class-level lab data if necessary.
        if self.lab_data is None:
            self.lab_data = load_lab_data()

    def setraw(self, raw, name=None):
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
                Name of the GFA that produced this raw data. Must be set to one of the values in gfa_names
                in order to lookup the correct master bias and dark images, and amplifier parameters, when
                these features are used.
        """
        if raw.ndim not in (2, 3):
            raise ValueError('raw data must be 2D or 3D.')
        raw_shape = (2 * self.nampy, 2 * self.nampx + 4 * self.nscan)
        if raw.shape[-2:] != raw_shape:
            raise ValueError('raw data has dimensions {0} but expected {1}.'.format(raw.shape[-2:], raw_shape))
        if raw.ndim == 2:
            raw = raw.reshape((1,) + raw_shape)
        self.nexp, ny, nx = raw.shape
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
                print('Ignoring {0} bad overscan pixels for {1} {2}.'.format(nbad.sum(), self.name, amp))
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