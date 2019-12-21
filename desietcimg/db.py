"""Interact with the online database at KPNO or its NERSC mirror.

Importing this module requires that the following packages are installed:
 - pyyaml
 - pandas
 - psycopg2
"""
import collections
import yaml
import datetime
import os.path

import numpy as np

import pandas as pd
import psycopg2


db_config = None

class DB(object):
    """Initialize a connection to the database.

    Parameters
    ----------
    config_path : str
        Path of yaml file containing connection parameters to use.
    """
    def __init__(self, config_name='db.yaml'):
        global db_config
        if db_config is None:
            if not os.path.exists(config_name):
                raise RuntimeError('Missing db config file: {0}.'.format(config_name))
            with open(config_name, 'r') as f:
                db_config = yaml.safe_load(f)
        self.conn = psycopg2.connect(**db_config)
    def query(self, sql, dates=None):
        return pd.read_sql(sql, self.conn, parse_dates=dates)
    def select(self, table, what, where=None, limit=10, order=None, dates=None):
        sql = f'select {what} from {table}'
        if where is not None:
            sql += f' where {where}'
        if order is not None:
            sql += f' order by {order}'
        if limit is not None:
            sql += f' limit {limit}'
        return self.query(sql, dates)


class Exposures(object):
    """Cacheing wrapper class for the exposure database.
    """
    def __init__(self, db, columns='*', cachesize=5000):
        # Run a test query.
        test = db.select('exposure.exposure', columns, limit=1)
        self.columns = list(test.columns)
        self.what = ','.join(self.columns)
        self.db = db
        self.cache = collections.OrderedDict()
        self.cachesize = cachesize

    def __call__(self, expid, what=None):
        if what is not None and what not in self.columns:
            raise ValueError(f'Invalid column name: "{what}".')
        if expid not in self.cache:
            row = self.db.select('exposure.exposure', self.what, where=f'id={expid}', limit=1)
            if row is None:
                raise ValueError('No such exposure id {0}.'.format(expid))
            # Cache the results.
            self.cache[expid] = row.values[0]
            # Trim the cache if necessary.
            while len(self.cache) > self.cachesize:
                self.cache.popitem(last=False)
            assert len(self.cache) <= self.cachesize
        values = self.cache[expid]
        if what is None:
            return values
        return values[self.columns.index(what)]


class NightTelemetry(object):
    """Lookup telemetry using a cache of local noon-noon results.
    """
    def __init__(self, db, tablename, columns='*', cachesize=10, timestamp='time_recorded', verbose=False):
        # Run a test query.
        test = db.select('telemetry.' + tablename, columns, limit=1)
        self.db = db
        self.cachesize = int(cachesize)
        self.tablename = tablename
        self.columns = list(test.columns)
        if timestamp not in self.columns:
            self.columns.append(timestamp)
        self.what = ','.join(self.columns)
        self.timestamp = timestamp
        if verbose:
            print(f'Initialized telemetry from {self.tablename} for {self.what}.')
        self.cache = collections.OrderedDict()
        self.MJD_epoch = pd.Timestamp('1858-11-17', tz='UTC')
        self.one_day = pd.Timedelta('1 days')

    def __call__(self, night, what=None, MJD=None):
        """Return the telemetry for a single night.

        Recently queried nights are cached so that repeated calls to this function for the
        same night only trigger one database query.

        Parameters
        ----------
        night : int
            Night specified as an integer YYYYMMDD.
        what : str or None
            Name of the single column to fetch or None to return all columns.
            When this is specified, the returned dataframe will have two columns:
            the requested one plus MJD.
        MJD : array or None
            Array of MJD values where the value of ``what`` will be tabulated using
            interpolation. Can be used to resample data to a uniform grid or
            downsample a large dataset. Must be accompanied by ``what`` specifying
            a numeric column and returns a 1D numpy array.

        Returns
        -------
        dataframe or numpy array
        """
        if what is not None and what not in self.columns:
            raise ValueError(f'Invalid column name "{what}". Pick from {self.what}.')
        if MJD is not None and what is None:
            raise ValueError(f'Must specify a column (what) with MJD values.')
        # Calculate local midnight on night = YYYYMMDD as midnight UTC + 31 hours (assuming local = UTC-7)
        try:
            midnight = datetime.datetime.strptime(str(night), '%Y%m%d') + datetime.timedelta(days=1, hours=7)
        except ValueError:
            raise ValueError(f'Badly formatted or invalid night: "{night}".')
        self.midnight = pd.Timestamp(midnight, tz='UTC')
        if night not in self.cache or MJD is not None:
            # Fetch data from local noon on YYYYMMDD until local noon the next day.
            tmin = self.midnight - pd.Timedelta(12, 'hours')
            tmax = self.midnight + pd.Timedelta(12, 'hours')
        if MJD is not None:
            MJD = np.asarray(MJD)
            # Check that the min MJD is within our range.
            timestamp = self.MJD_epoch + MJD.min() * self.one_day
            if timestamp < tmin or timestamp > tmax:
                raise ValueError(f'MJD {MJD.min()} ({timestamp}) not in night {night}.')
            # Check that the max MJD is within our range.
            timestamp = self.MJD_epoch + MJD.max() * self.one_day
            if timestamp < tmin or timestamp > tmax:
                raise ValueError(f'MJD {MJD.max()} ({timestamp}) not in night {night}.')
        if night not in self.cache:
            # Fetch the results.
            results = self.db.select(
                self.tablename, self.what, limit=None,
                where=f"{self.timestamp}>=TIMESTAMP '{tmin}' and {self.timestamp}<=TIMESTAMP '{tmax}'")
            # Convert the timestamp column to MJD.
            results['MJD'] = (results[self.timestamp] - self.MJD_epoch) / self.one_day
            # Cache the results.
            self.cache[night] = results
            # Trim the cache if necessary.
            while len(self.cache) > self.cachesize:
                self.cache.popitem(last=False)
            assert len(self.cache) <= self.cachesize
        # Fetched the cached results.
        results = self.cache[night]
        if what is None:
            return results
        # Select the specified column (in addition to MJD).
        results = results[['MJD', what]]
        if MJD is None:
            return results
        # Interpolate to the specified time (assuming "what" is numeric).
        dtype = results[what].dtype
        if not np.issubdtype(dtype, np.number):
            raise ValueError(f'Nearest neighbor lookup not implemented yet for dtype "{dtype}".')
        return np.interp(MJD, results['MJD'], results[what])
