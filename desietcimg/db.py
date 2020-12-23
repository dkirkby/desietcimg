"""Interact with the online database at KPNO or its NERSC mirror.

Importing this module requires that the pandas package is installed.
Creating a DB() instance will either require that pyyaml and psycopg2
are installed (for a direct connection), or else that requests is installed
(for an indirect http connection).
"""
import collections
import datetime
import os.path
import logging
import io

import numpy as np

import pandas as pd

try:
    import requests
except ImportError:
    # We will flag this later if it matters.
    pass


class DB(object):
    """Initialize a connection to the database.

    To force a direct connection using pyscopg2, set ``http_fallback``
    to ``False``. To force an indirect http connection using requests,
    set ``config_name`` to ``None``.  By default, will attempt a
    direct connection then fall back to an indirect connection.

    Direct connection parameters are stored in the SiteLite package.

    An indirect connection reads authentication credentials from
    your ~/.netrc file. Refer to this internal trac page for details:
    https://desi.lbl.gov/trac/wiki/Computing/AccessNerscData#ProgrammaticAccess

    Parameters
    ----------
    config_path : str
        Path of yaml file containing direct connection parameters to use.
    http_fallback : bool
        Use an indirect http connection when a direct connection fails
        if True.
    """
    def __init__(self, config_name='db.yaml', http_fallback=True):
        self.method = 'indirect'
        if os.path.exists(config_name):
            # Try a direct connection.
            try:
                import yaml
            except ImportError:
                raise RuntimeError('The pyyaml package is not installed.')
            with open(config_name, 'r') as f:
                db_config = yaml.safe_load(f)
            try:
                import psycopg2
                self.conn = psycopg2.connect(**db_config)
                self.method = 'direct'
            except ImportError:
                if not http_fallback:
                    raise RuntimeError('The psycopg2 package is not installed.')
            except Exception as e:
                if not http_fallback:
                    raise RuntimeError(f'Unable to establish a database connection:\n{e}')
        if self.method == 'indirect' and http_fallback:
            try:
                import requests
            except ImportError:
                raise RuntimeError('The requests package is not installed.')
        logging.info(f'Established {self.method} database connection.')
    def query(self, sql, dates=None):
        if self.method != 'direct':
            raise RuntimeError('SQL only supported for a direct connection.')
        return pd.read_sql(sql, self.conn, parse_dates=dates)
    def select(self, table, what, where=None, limit=10, order=None, dates=None):
        if self.method == 'direct':
            sql = f'select {what} from {table}'
            if where is not None:
                sql += f' where {where}'
            if order is not None:
                sql += f' order by {order}'
            if limit is not None:
                sql += f' limit {limit}'
            return self.query(sql, dates)
        else:
            url = 'https://replicator.desi.lbl.gov/QE/DESI/app/query'
            params = dict(
                dbname='desi',
                tables=table,
                columns=what,
                maxrows=limit,
                output_type='text,', # specify CSV
                #dsid=3, # is this necessary? what is it??
            )
            if where:
                params['wheres'] = where
            if order:
                params['orders'] = order
            logging.debug(f'indirect query params: {params}')
            req = requests.get(url, params=params)
            if req.status_code != requests.codes.ok:
                logging.warning(f'HTTP status {req.status_code}')
                req.raise_for_status()
            df = pd.read_csv(io.StringIO(req.text), parse_dates=dates)
            # Note that the server adds a comma to the end of each line, including the header,
            # so there will be a final un-named column with empty values.  Drop it here.
            return df.iloc[:, :-1]

class Exposures(object):
    """Cacheing wrapper class for the exposure database.
    """
    def __init__(self, db, columns='*', cachesize=5000):
        # Run a test query.
        test = db.select('exposure.exposure', columns, limit=1)
        self.columns = list(test.columns)
        logging.debug(f'exposure table columns: {self.columns}')
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
