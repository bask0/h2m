
from torch.utils.data import Dataset
import numpy as np
import os
import shutil
import json
from collections import OrderedDict
import dateutil
import datetime
import copy
from typing import Union, Iterable, Tuple, List, Dict, Any, Callable

import torch
import re

import pandas as pd
import xarray as xr
import zarr


class SpatialData(object):
    """Characterizes a NetCDF file, checks some requirements and provides reading utilities.

    Note: This class can represent datasets with or without time dimension. If the dataset has no
    time dimension, the argument 'hastime' can be set to False and the other parameters related to
    time (timename, timebinname, hastimebin) take no effect.

    Parameters
    ----------
    path
        Path to NetCDF file, string.
    varname
        Variable name, string, default is 'val'.
    varname_new
        Optional new variable name.
    lonname
        Name of longitude dimentions, string, default check is 'lon' and 'longitude'.
    timename
        Name of time dimentions, string, default is 'time'.
    timebinname
        Name of timebinname dimentions, string, default is 'time_bins'.
    hastime
        Whether dataset has a time dimension, bool, default is True.
    hastimebin
        Whether dataset has a time bin dimension, bool, default is False.
        A time bin dimension defines the start and stop time of the periode that
        is represented by each values along time dimension.
    istarget
        Wheter the dataset is a target (True) or a feature (False).
    unit
        Data unit, e.g. 'mm / d'.
    dtype
        Optional dtype to which the data is cast when calling 'get_writeable(...)'.
    """

    def __init__(
            self,
            path: str,
            varname: str = 'val',
            varname_new: str = '',
            latname: str = 'infer',
            lonname: str = 'infer',
            timename: str = 'time',
            timebinname: str = 'time_bounds',
            hastime: bool = True,
            hastimebin: bool = False,
            istarget: bool = False,
            unit: str = '',
            dtype: type = np.float32) -> None:

        self._path = path
        self._varname = varname
        self._varname_new = varname_new
        self._latname = latname
        self._lonname = lonname
        self._timename = timename
        self._timebinname = timebinname
        self._hastime = hastime
        self._hastimebin = hastimebin
        self._istarget = istarget
        self._unit = unit
        self._dtype = dtype

        if self._hastimebin:
            assert self._hastime, '`hastimebin` cannnot be True as `hastime` is False.'

        if self._varname_new == '':
            self._varname_new = self._varname

        self._consistency_check()

        with xr.open_dataset(self._path) as d:
            self._lat = d[self._latname].data
            self._lon = d[self._lonname].data
            if self._hastime:
                self._time = d[self._timename].data
                self._time_d = (pd.DatetimeIndex(self._time) -
                                pd.datetime(1970, 1, 1)).days.values

                vardim = np.setdiff1d(
                    d[self._varname].dims,
                    (self._latname, self._lonname, self._timename))

            else:
                self._time = None
                self._time_d = None

                vardim = np.setdiff1d(
                    d[self._varname].dims,
                    (self._latname, self._lonname))

            if len(vardim) == 1:
                self._vardim = vardim[0]
                self._vardim_len = len(d[self._vardim])
            elif len(vardim) == 0:
                self._vardim = None
                self._vardim_len = None
            else:
                raise ValueError(
                    'The variable dimensions do not match requirements.')

            if self._hastimebin:
                self._timebin = d[self._timebinname].data
                self._timebin_d = np.array(
                    [(pd.DatetimeIndex(self._timebin[:, col]) - pd.datetime(1970, 1, 1)).days.values
                     for col in range(2)]).T
            else:
                self._timebin = None
                self._timebin_d = None

        self._nlat = len(self._lat)
        self._nlon = len(self._lon)
        self._ntime = len(self._time) if self._hastime else None
        self._ntimebin = self._timebin.shape[0] if self._hastimebin else None

        if self._hastimebin:
            assert self._ntime == self._ntimebin, (
                'Number of time bins ({}) must be equal to number of time steps ({}).'.format(
                    self._ntimebin, self._ntime))

    def get_writeable(
            self,
            scale_factor: int = 1) -> 'SpatialData':
        """Get a writable version of the datasets.

        Parameters
        ----------
        scale_factor
            The scale factor of the dataset that is used to rearrange the spatial dimension
            of the dataset.

        Returns
        ----------
        A 'clean' version of the dataset.

        """

        if scale_factor == 1:
            var = self.var
        else:
            var = self._rescale(scale_factor)

        if self._dtype is not None:
            var = var.astype(self._dtype)

        # Create dataset.
        ds_new = xr.Dataset()
        ds_new['data'] = var

        # Put attributes back.
        ds_new.attrs = self.ds.attrs

        # If the dataset has a timebins, put them back.
        if self.hastimebin:
            ds_new[self.timebinname] = self.ds[self.timebinname]

        if self.hastime:
            ds_new = ds_new.rename({
                self.timename: 'time'
            })

        if self.hastimebin:
            ds_new = ds_new.rename({
                self.timebinname: 'timebin'
            })

        # Some datasets store data with ascending latitude (-90, ..., 90). We stick
        # to the convention of descending latitudes (90, ..., -90) as this puts
        # north-west pixel at [0, 0] index of array when writing as zarr file.
        if ds_new.lat[0].data < ds_new.lat[1].data:
            ds_new = ds_new.reindex(lat=list(reversed(ds_new.lat)))

        return ds_new

    def get_time_coverage(
            self,
            time_range: Union[slice, Iterable[int]],
            freq: str = 'D',
            warmup: str = '0D') -> Tuple[Tuple[int, int], List[List[int]]]:
        '''Get handlers for time coverage based on reference time range.

        This method returns lower and upper indices for the dataset's time dimension that fully
        covers the given 'time_range', defining a lower and upper bounds of a date range. If the
        dataset has time bins (a.k.a. bounds), indices for the time bins fully covering the given
        'time_range' are computed. Also, for datasets with time bins, the function returns a lower
        and upper bound per time step that covers the 'time_range'. These bounds indicate the indices
        of the date range ('time_range' bounds with 'freq' as step size) that are covered by a
        time bin. This can be used to aggregate high-resolution predictions to the lower resolution
        of iregulary spaced time bins.

        Parameters
        ----------
        time_range
            A slice (slice(lower, upper)) object or an iterable of length 2 defining the lower and upper
            time bounds to select, must be interpretable by dateutil.parser.parse(...), e.g. '2001-01-01'.
        freq
            The resolution of the reference dates, i.e. the base date resolution the time bins are translated
            to. See pandas.date_range for valid requencies, default is 'D' for day.
        warmup
            Warmup period, encoded as a string of a number followed by 'D' for days or 'Y' for years, e.g.
            '1Y'. The warmup period is added from lower time bound for training time-series. Only applies
            if dataset 'self._istarget' is False.

        Returns
        ----------
        A tuple with the lower and upper indices of this dataset that matches the reference dates and a list of
        indices that define lower and upper indices of each time-step for datasets with time bins.
        '''

        if not self.hastime:
            raise ValueError(
                'Cannot use this method as this SpatialData has no time dimension.')

        # Parse time_range.
        if isinstance(time_range, slice):
            date_start_ = time_range.start
            date_stop_ = time_range.stop
        elif hasattr(time_range, '__iter__'):
            if len(time_range) != 2:
                raise ValueError('Argument ```time_range``` must be of type slice or an '
                                 'iterable of length 2.')
            date_start_ = time_range[0]
            date_stop_ = time_range[1]
        else:
            raise ValueError('Argument ```time_range``` must either be of type ```slice``` '
                             'or an iterable of length 2.')

        # Parse warmup.
        if not self._istarget:
            warmup = '0D'
        warmup_steps = np.int(warmup[:-1])
        warmup_freq = warmup[-1]

        date_start = dateutil.parser.parse(date_start_)
        date_stop = dateutil.parser.parse(date_stop_)

        if warmup_freq == 'D':
            warmup_steps = warmup_steps
            warmup_start_date = date_start + \
                datetime.timedelta(days=warmup_steps)
        elif warmup_freq == 'Y':
            warmup_start_date = datetime.datetime(date_start.year + warmup_steps, date_start.month,
                                                  date_start.day, date_start.minute, date_start.microsecond)
            warmup_steps = len(pd.date_range(
                date_start, warmup_start_date)) - 1
        else:
            raise ValueError('Argument warmup={} mut be an string combination of an integer '
                             'followed by `D` (days) or `Y` (years).'.format(warmup))

        # From here on, we work with days since 1970 instead of dates.
        ref_date = pd.datetime(1970, 1, 1)
        start = (date_start - ref_date).days
        stop = (date_stop - ref_date).days
        start_pred = start + warmup_steps

        if self.hastimebin:
            dates = self.timebin_d
        else:
            dates = self.time_d

        ndim = dates.ndim
        # If one dimension, dates are single time stamps.
        if ndim == 1:
            dates_lower = dates_upper = dates
        # If two dimensions, dates are time bounds.
        elif ndim == 2:
            dates_lower = dates[:, 0]
            dates_upper = dates[:, 1]
        else:
            raise ValueError('```self.dates``` must have 1 or 2 dimentions, '
                             'but has {} dimensions.'.format(ndim))

        # Get lower bound index of element that is covered by the time_slice.
        lower_index = np.argwhere(dates_lower >= start_pred)
        if len(lower_index) == 0:
            raise ValueError('Lower time bound ({}) is outside data coverage ({} ---- {}), resulting in '
                             'an empty dataset.'.format(date_start, dates_lower[0], dates_upper[-1]))
        else:
            lower_index = lower_index[0][0]
        upper_index = np.argwhere(dates_upper <= stop)

        # Get upper bound index of element that is covered by the time_slice.
        if len(upper_index) == 0:
            raise ValueError('Upper time bound ({}) is outside data coverage ({} ---- {}), resulting in '
                             'an empty dataset.'.format(date_stop, dates_lower[0], dates_upper[-1]))
        else:
            # +1 as python slicing does not include last index.
            upper_index = upper_index[-1][0] + 1
        if not upper_index > lower_index:
            raise ValueError('The time subset is empty.')

        # Reference index range.
        ref_idx = np.arange(start, stop + 1)

        # For SpatialData with time bins: Lower / upper indices that are covered by each time bin.
        if self.hastimebin:
            # Get time bins that are covered by 'time_slice'.
            covered_bins = dates[lower_index:upper_index]

            # Empty to store lower and upper bounds indices.
            ref_bins = np.zeros(covered_bins.shape, dtype=np.int)

            # Iterate the time bins and get lower upper bounds indices.
            for i, cb in enumerate(covered_bins):
                # Lower index.
                a = int(np.argwhere(ref_idx >= cb[0])[0])
                # Upper index, +1 as used to index python style.
                b = int(np.argwhere(ref_idx <= cb[1])[-1]) + 1
                ref_bins[i, :] = a, b

            # Convert to list as required by json serialization.
            ref_bins = ref_bins.tolist()
        else:
            # Return slice relative to entire slice if SpatialData has no time bins.
            covered_bins = dates[lower_index:upper_index]
            ref_bin_start = np.argwhere(ref_idx == start_pred)[0][0]
            ref_bin_stop = np.argwhere(ref_idx == stop)[0][0] + 1
            ref_bins = [int(ref_bin_start), int(ref_bin_stop)]

        return (int(lower_index), int(upper_index)), ref_bins

    def _rescale(self, scale_factor: int) -> xr.DataArray:
        """Rescale xr.Dataset to match reference spatial resolution.

        The datast is reshaped to match a reference resolution by adding blocks in
        the spacial dimension of the size scale_factor x scale_factoe to a new dimension.

        E.g. A dataset with dimensions
            -> (time x lat x lon) = (10 x 12 x 12)
        and a scale factor of 3 would result in a new dataset of size
            -> (time x lat x lon x lat_index x lon_index) = (10, 4, 4, 3, 3).

        Paramters
        ----------
        scale_factor
            The factor by which this dataset is larger as a reference resolution, an integer.

        Returns
        ----------
            Returns a rescaled xr.DataArray.
        """
        nlat_new = self.nlat / scale_factor
        nlon_new = self.nlon / scale_factor
        assert nlat_new % 1.0 == 0.0, 'The dataset\'s dimension \'lat\' has size {}, '\
            'must be fully dividable by the argument \'scale_factor\' ({}).'.format(
                self.nlat, scale_factor)
        assert nlon_new % 1.0 == 0.0, 'The dataset\'s dimension \'lon\' has size {}, '\
            'must be fully dividable by the argument \'scale_factor\' ({}).'.format(
                self.nlon, scale_factor)
        nlat_new = int(nlat_new)
        nlon_new = int(nlon_new)
        # Coordinates needed to build new xr.Dataset.
        coords = [
            xr.IndexVariable('lat_index', np.arange(scale_factor)),
            xr.IndexVariable('lon_index', np.arange(scale_factor)),
            xr.IndexVariable(
                'var', self.var[self._vardim]) if self._vardim is not None else 0,
            xr.IndexVariable('lat',
                             np.linspace(90 - 180 / (nlat_new) / 2, -90 + 180 / (nlat_new) / 2, nlat_new)),
            xr.IndexVariable('lon',
                             np.linspace(-180 + (360 / nlon_new / 2), 180 - (360 / nlon_new / 2), nlon_new))
        ]
        if self._vardim is None:
            coords.pop(2)

        with self.var as ds:
            # Optionally add time dimension.
            if self.hastime:
                coords = [ds.time] + coords
                rs = (
                    self.ntime,
                    nlat_new,
                    scale_factor,
                    nlon_new,
                    scale_factor)
                tr = (0, 2, 4, 1, 3)
                # Reshape data.
                data = ds.data.reshape(*rs).transpose(*tr)
            else:
                if self._vardim is not None:
                    rs = (
                        self._vardim_len,
                        nlat_new,
                        scale_factor,
                        nlon_new,
                        scale_factor)
                    tr = (2, 4, 0, 1, 3)
                else:
                    rs = (
                        nlat_new,
                        scale_factor,
                        nlon_new,
                        scale_factor)
                    tr = (1, 3, 0, 2)
                # Reshape data.
                data = ds.data.reshape(*rs).transpose(*tr)

            # Create DataArray.
            var = xr.DataArray(
                data,
                coords=coords
            )
            # Put attributes back.
            var.attrs = self.var.attrs

        return var

    def _consistency_check(self) -> None:
        """Check arguments for consistency (dimensions, dtypes etc.)."""
        # Check path.
        assert os.path.isfile(
            self._path), 'Not a valid file path:\n{}'.format(self._path)

        # Check variables.
        with xr.open_dataset(self._path) as d:

            if self._varname not in d.data_vars:
                raise Exception('Variable \'{}\' not found in dataset. {}'.format(
                    self._varname, d.data_vars))

            # Infer lon lat name.
            if self._latname == 'infer':
                if 'lat' in d.dims:
                    self._latname = 'lat'
                elif 'latitude' in d.dims:
                    self._latname = 'latitude'
                else:
                    raise Exception('Latitude name could not be infered. Dataset dimensions: {}\n{}'.format(
                        ' x '.join([*d.dims]), self._path))

            if self._lonname == 'infer':
                if 'lon' in d.dims:
                    self._lonname = 'lon'
                elif 'longitude' in d.dims:
                    self._lonname = 'longitude'
                else:
                    raise Exception('Longitude name could not be infered. Dataset dimensions: {}\n{}'.format(
                        ' x '.join([*d.dims]), self._path))

            # Check if dimensions are present.
            assert self._latname in d.dims, (
                'Latitude name \'{}\' could not found in dataset*. Dataset dimensions: {}\n* {}'.format(
                    self._latname, ' x '.join([*d.dims]), self._path))
            assert self._latname in d.dims, (
                'Latitude name \'{}\' could not found in dataset*. Dataset dimensions: {}\n* {}'.format(
                    self._latname, ' x '.join([*d.dims]), self._path))
            assert self._lonname in d.dims, (
                'Longitude name \'{}\' could not found in dataset*. Dataset dimensions: {}\n* {}'.format(
                    self._lonname, ' x '.join([*d.dims]), self._path))
            if self._hastime:
                assert self._timename in d.dims, (
                    'Time name \'{}\' could not found in dataset*. Dataset dimensions: {}\n* {}'.format(
                        self._timename, ' x '.join([*d.dims]), self._path))
            if self._hastime & self._hastimebin:
                assert self._timebinname in d.data_vars, (
                    'Time bin name \'{}\' could not found in dataset*. Dataset variables: {}\n* {}'.format(
                        self._timebinname, ' x '.join([*d.data_vars]), self._path))

            # Check dimension order and variable dimensionality.
            if self._hastime:
                assert d[self._varname].dims[:3] == (self._timename, self._latname, self._lonname), (
                    'Dimension order is {} but should be {}.'.format(
                        d[self._varname].dims, (self._timename, self._latname, self._lonname)))
                assert d[self._varname].ndim == 3, 'A dataset variable with time dimension must have '\
                    '3 dimensions, but has {}.'.format(d[self._varname].ndim)
            else:
                dims = d[self._varname].dims
                if len(dims) == 3:
                    dims = dims[1:]
                else:
                    dims = dims
                assert dims == (self._latname, self._lonname), 'Dimension order is {} '\
                    'but should be (optional[var], {}, {}).'.format(
                        d[self._varname].dims, self._latname, self._lonname)
                assert d[self._varname].ndim in [2, 3], 'A dataset variable without time dimension must have '\
                    '2 or 3 (if multiple variables) dimensions, but has {}.'.format(
                        d[self._varname].ndim)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = '>>> {:15} >>> '.format(self._varname_new[:15])
        s += ' x '.join(['{} ({})'.format(dim, len(self.var[dim]))
                         for dim in self.var.dims])
        s += '  >>> ...{}'.format(self._path[-60:])
        return s

    def __eq__(self, other: 'SpatialData') -> bool:
        if self.__class__ == other.__class__:
            if str(self) == str(other):
                return True
            else:
                return False
        else:
            raise TypeError('Comparing object is not of the same type.')

    @property
    def nlat(self) -> int:
        return self._nlat

    @property
    def nlon(self) -> int:
        return self._nlon

    @property
    def ntime(self) -> int:
        return self._ntime

    @property
    def hastime(self) -> bool:
        return self._hastime

    @property
    def hastimebin(self) -> bool:
        return self._hastimebin

    @property
    def istarget(self) -> bool:
        return self._istarget

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def time_d(self) -> np.ndarray:
        return self._time_d

    @property
    def timebin(self) -> np.ndarray:
        return self._timebin

    @property
    def timebin_d(self) -> np.ndarray:
        return self._timebin_d

    @property
    def varname(self) -> str:
        return self._varname

    @property
    def varname_new(self) -> str:
        return self._varname_new

    @property
    def latname(self) -> str:
        return self._latname

    @property
    def lonname(self) -> str:
        return self._lonname

    @property
    def timename(self) -> str:
        return self._timename

    @property
    def timebinname(self) -> str:
        return self._timebinname

    @property
    def path(self) -> str:
        return self._path

    @property
    def dtype(self) -> type:
        if self._dtype is None:
            return self.getvar('mask').dtype
        else:
            return self._dtype

    @property
    def shape(self) -> Tuple[int]:
        return self.var.shape

    @property
    def ds(self) -> xr.Dataset:
        ds = xr.open_dataset(self._path)
        ds = ds.rename({self.varname: self.varname_new,
                        self.latname: 'lat',
                        self.lonname: 'lon'})
        return ds

    @property
    def var(self) -> xr.DataArray:
        var = self.ds[self.varname_new]
        return var

    @property
    def unit(self) -> str:
        return self._unit


class SampleFormatter(object):
    """Defining format of data returned from Sampler.

    Config file requirements:
        JSON file containing information on what data to read from a Bucket
        and how to structure the returned arrays.

        Fields:
            "hastime": bool, whether the dataset has a dime dimension.
            "convolution": bool, whether the dataset has a higher resolution as the reference resolution.
            "istarget":  bool, wheterh the dataset is a targrt variable.
            "hastimebin": bool, wheterh the dataset has time bins (defining lower/upper time
                bound for eachobservation).
            "preserve_rel": optional, whether to normalize each variable in the dataset individually (False, default), or to
                normalize all variables together to preserve relative values (True). Only applies for non-temporal data.

        Example:

        {
            "description": "Features and target datasets for hydrological model.",
            "datasets": {
                "tair": {
                    "hastime": true,
                    "convolution": false,
                    "istarget": false,
                    "hastimebin": false
                },
                "tws": {
                    "hastime": true,
                    "convolution": false,
                    "istarget": true,
                    "hastimebin": true
                },
                "globland": {
                    "hastime": false,
                    "convolution": true,
                    "istarget": false,
                    "hastimebin": false,
                    "preserve_rel": true
                }
            },
            "time_slices": {
                "train": ["2001-01-01", "2008-12-31"],
                "valid": ["2008-01-01", "2014-12-31"],
                "test":  ["2008-01-01", "2014-12-31"]
            },
            "warmup": "1Y"
        }

    Parameters
    ----------
    config_file
        Path do a JSON config file, see 'Config file requirements' in class documentation.
    """

    def __init__(self, config_file: str) -> None:

        with open(config_file) as file:
            cf = json.load(file)

        self.cf_path = config_file
        self.dict = cf
        self.description = cf['description']
        self.datasets = self.dict['datasets']
        self._validate_structure()

        self.ndatasets = len(self.datasets)

        self.features, self.features_dynamic, self.features_static, self.targets = self._get_features_and_targets()

    def _validate_structure(self) -> None:
        """Validate configuration file structure."""

        # Check top-level required keys.
        required = ['description', 'datasets', 'time_slices', 'warmup']
        check_keys = list(self.dict.keys())
        for key in required:
            assert key in check_keys,\
                'Could not find required key \'{}\' in \'config_file\': {}'.format(
                    key, self.cf_path)

        # Check required keys of datasets.
        required = ['hastime', 'hastimebin', 'convolution', 'istarget']
        for dataset in list(self.datasets.keys()):
            ds = self.datasets[dataset]
            check_keys = list(ds.keys())
            for key in required:
                assert key in check_keys,\
                    'Could not find required key \'{}\' for the dataset \'{}\' in '\
                    '\'config_file\': {}'.format(key, dataset, self.cf_path)

        # Check 'preserve_rel' key of datasets.
        for dataset in list(self.datasets.keys()):
            ds = self.datasets[dataset]
            # If 'preserve_rel' not set, set it to False.
            if 'preserve_rel' not in ds:
                self.datasets[dataset]['preserve_rel'] = False
            else:
                preserve_rel = self.datasets[dataset]['preserve_rel']
                if preserve_rel not in [True, False]:
                    raise ValueError(
                        f'Dataset `{dataset}`: Key `preserve_rel` must either not be passed ',
                        f'or set to `true` or `false`, but is {preserve_rel}.')

            check_keys = list(ds.keys())
            for key in required:
                assert key in check_keys,\
                    'Could not find required key \'{}\' for the dataset \'{}\' in '\
                    '\'config_file\': {}'.format(key, dataset, self.cf_path)

        # Check time_slices.
        required = ['train', 'valid', 'test']
        for s in required:
            assert s in self.dict['time_slices'].keys(),\
                'The field \'{}\' in \'time_slices\' of the \'SampleFormatter\' config file does not '\
                'exist. Config file location: {}'.format(s, self.cf_path)
            assert (len(self.dict['time_slices'][s]) == 2) and (not isinstance(self.dict['time_slices'][s], str)),\
                'The field \'{}\' in \'time_slices\' must be a list of two elements. Config file '\
                'location: {}'.format(s, self.cf_path)

        # Check warmup.
        v = self.dict['warmup']
        assert isinstance(v, str), 'The value \'\"{}\"\' in field \'warmup\' is not a string, '\
            'e.g. "1D" or "2Y". Config file location: {}'.format(
            self.dict['warmup'], self.cf_path)

    def _get_features_and_targets(self) -> Tuple[Dict[str, str]]:
        features = []
        features_dynamic = []
        features_static = []
        targets = []
        for ds_name, ds in self.datasets.items():
            # Check if the stack is a target.
            istarget = ds['istarget']
            hastime = ds['hastime']

            if istarget:
                # Add datasets in given stack to dict.
                targets.append(ds_name)
            else:
                # Add datasets in given stack to dict.
                features.append(ds_name)
                if hastime:
                    features_dynamic.append(ds_name)
                else:
                    features_static.append(ds_name)

        return features, features_dynamic, features_static, targets

    def __getitem__(self, key):
        return self.dict[key]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return json.dumps(self.dict, indent=4, sort_keys=True)

    @property
    def num_features(self):
        return len(self.features)

    @property
    def num_featues_dynamic(self):
        return len(self.features_dynamic)

    @property
    def num_features_static(self):
        return len(self.features_static)

    @property
    def num_targets(self):
        return len(self.targets)


class Bucket(Dataset):
    """Represents a data bucket (data structure to read sampels from).

    A bucket is a data structure optimized to extract single samples.The underlying
    data structure, Zarr ('chunked, compressed, N-dimensional arrays'), enables
    multihead reading. The sampling capabilities build upon Pytorch's Dataset /
    Dataloader framework

    Creation
    ----------------
    To create a new Bucket, a mask of type SpatialData is needed:
    >> mask = SpatialData('./my/mask.nc',varname='mask', varname_new='land_mask', hastime=False, dtype=np.int8)

    Next, the bucket is created:
    >> b = Bucket('./my/bucket.zarr', nlat=180, nlon=360, mask=mask, overwrite=True, chunk_size=10,
        sample_formatter_path='./test_config.json')

    Note that a sample formatter file needs to be passed for creation of a new Bucket. This is a current
    limitation, as a new Bucket needs to be created for different sampling format. A workaround is to manually
    set a new SampleFormatter that has the same values for time_slices and warmup. A deviation from
    this leads to unexpected behaviour.

    Then, datasets can be added to the bucket.

    >> tair = SpatialData('./my/variable.nc', varname='tair', varname_new='tair')
    >> rn = SpatialData('/Net/Groups/BGI/people/bkraft/01_hydro_dl/data/1d/daily/rn.2001_2012.nc',
                   varname='Rn', varname_new='rn')
                   varname='val', varname_new='swe', hastimebin=True, timebinname='time_bnds', istarget=True)
    >> tws = SpatialData('/./my/target.nc', varname='lwe_thickness', varname_new='lwe',
        hastimebin=True, timebinname='time_bounds', istarget=True)
    >> b.isinstance(other,tair, tws)

    The datasets are written to the Bucket and according attributes defining time slices and bins (to
    match features and potentially inequally spaced targets) are added to the data.

    Sampling setup
    ----------------

    Next, the spatial sampling for the sampling sets are defined; the values correspond to
    the mask, e.g. the training sampler would sample from locations where the mask has values 1 or 2.

    >> b.set_space_ids(training_ids=[1, 2], valid_ids=[3], test_ids=[4])

    The temporal sampling is set in the SampleFormatter. Now, samplers are derived from the Bucket.

    >> train_sampler = b.get_sampler('train')
    >> valid_sampler = b.get_sampler('valid')
    >> test_sampler = b.get_sampler('test')

    Sampling
    ----------------
    We can now pass the sampler to Pytorch's DataLoader and use all of its functionalities.

    >> dl_train = DataLoader(train_sampler, batch_size=10, shuffle=True, num_workers=0)

    >> for i, batch in enumerate(dl_train):
         #print('{:2d}: {}'.format(i, batch))
         batch = batch['dynamic']['data']

    A batch contains time bins for each target variable that has time bins indicating what
    indices of the feature datasets corresponds to the target.

    Note: With the current implementation, the written files time dimension cannot be
    interpreted as such when being opened with zarr.open..., but with xarray.open_zarr.
    The current workaround is to read all dimensions with xarray and the (fast and
    parallel) data access is done with zarr.

    Parameters
    ----------
    path
        Path (store) to root (top directory) of bucket.
    nlat, nlon
        Dimension size of latitude and longitude, integer.
    chunk_size:
        An integer indicating the chunk size along lat / lon. Larger values lead
        to slower reading, too small values lead to long writing times.
    mask
        A mask with dimentions lat & lon of type SpatialData that indicates sampling
        set positions, e.g. unique values 0, 1, 2 represent: 0 = non-valid (convention).
        1 & 2: used to switch sets by determining pixels to sample from.
    overwrite
        If True, existing bucket with all contained datasets will be REMOVED.
        Else, data is added to the existing structure.
    use_msc_spinup
        If `True`, use repeated mean seasonal cycle as spinup data instead of random
        years.
    sample_formatter_path
        A path to a SampleFormatter defining what data is read from
        the bucket and how it is structured.
    dtype (default: np.float32)
        The "global" data type, all datasets are converted to thiosdtype.

    """

    def __init__(
            self,
            path: str,
            sample_formatter_path: str,
            nlat: int = None,
            nlon: int = None,
            chunk_size: int = None,
            mask: SpatialData = None,
            use_msc_spinup: bool = False,
            overwrite: bool = False,
            read_only: bool = True,
            dtype: type = np.float32) -> None:

        self._path = path
        self._nlat = nlat
        self._nlon = nlon
        self._chunk_size = chunk_size
        self._read_only = read_only
        if read_only:
            overwrite = False
        self._overwrite = overwrite
        self._dtype = dtype

        self._sample_formatter = SampleFormatter(sample_formatter_path)

        self.use_msc_spinup = use_msc_spinup

        # Write mode:
        if self._overwrite:
            # 'w' means create (overwrite if exists).
            open_mode = 'w'
        else:
            # 'a'  means read/write (create if doesn't exist).
            open_mode = 'a'

        if read_only:
            open_mode = 'r'

        # Create root group for Bucket.
        self.root = zarr.open_group(self._path, mode=open_mode)

        # Initialize metadata.
        self._initialize()

        # If mask is passed, add it to dataset (overwrite if exists).
        if mask is None:
            assert 'mask' in self.root, 'As the bucket does not contain a mask, you need to pass '\
                'arg \'mask\'.'
        else:
            self.set_mask(mask)

        # Initilizing sample attributes. These need to be set using respective functions before
        # Sampling is possible.
        self.space_indices = None  # self.set_space_ids(...)
        self.time_set = None  # Derived from SampleFormatter
        self.time_slices = None
        self._sample_set = None

        # This represents the time dimension of the features, which all must be equal. Will
        # only be set for derivatived sapler of bucket from ``sampler=bucket.get_sampler()``.
        self._feature_time = None

        self._mean_std = None
        self._datasets = None

    def set_space_ids(
            self,
            cv_mask_indices: Dict[str, Iterable[int]],
            bounds=None) -> None:
        """Spatial set indices to sample from.

        Parameters
        ----------
        cv_mask_indices
            A dict with the keys 'training', 'eval', 'test'. The valjues are the IDs for the respective sets,
            cooresponding to the mask (indicating locations to sample from, cooresponding to IDs).
        bounds
            The AOI bounds, an iterable of four numeric values: (min lat, max lat, min lon, max lon). If passed,
            the area is reduced to the specified extent. If this is not ``None``, no check is done if
            ``cv_mask_indices`` exist in mask.

        """

        mask = self.getxrvar('mask')

        if bounds is not None:
            # Reduce AOI size.
            mask = mask.where((
                mask.lat > bounds[0]) & (
                mask.lat < bounds[1]) & (
                mask.lon > bounds[2]) & (
                mask.lon < bounds[3]
            ), 0, drop=False)
            print(
                f'AOI: {bounds[0]} < lat < {bounds[1]}, {bounds[2]} < lon < {bounds[3]}')

        unique_values = np.unique(mask)

        for k in ['train', 'valid', 'test']:
            if k not in cv_mask_indices.keys():
                raise ValueError(
                    f'Key ``{k}`` not found in ``cv_mask_indices``.')

        if bounds is None:
            for k, v in cv_mask_indices.items():
                for i in v:
                    if i not in unique_values:
                        raise ValueError(
                            f'Mask value ``{i}`` does not exist. Possible choices: {unique_values} (0 are masked '
                            'values by convention, only use it if you know for sure you want to sample from 0).')

        self.space_indices = {
            'train': np.argwhere(np.isin(mask, cv_mask_indices['train'])),
            'valid': np.argwhere(np.isin(mask, cv_mask_indices['valid'])),
            'test': np.argwhere(np.isin(mask, cv_mask_indices['test'])),
            'all': np.argwhere(mask.values > 0)
        }

    def set_cv_fold_split(self, fold, latoffset=0, lonoffset=0, n_sets=8):
        state = np.random.RandomState(2)
        mask = self.getxrvar('mask')
        grid = self.get_cv_grid(
            mask, 2, latoffset=latoffset, lonoffset=lonoffset,) * (mask > 0) > 0
        indices = state.permutation(np.argwhere((grid > 0).values))
        sets = np.array_split(indices, n_sets, axis=0)

        # For the final predictions, we need the full mask (not gridded).
        grid_all = self.get_cv_grid(
            mask, 2, latoffset=latoffset, lonoffset=lonoffset, no_mask=True) * (mask > 0) > 0
        indices_all = state.permutation(np.argwhere((grid_all > 0).values))
        sets_all = np.array_split(indices_all, n_sets, axis=0)

        train_folds = np.setdiff1d(np.arange(n_sets - 1), fold)

        if fold > (n_sets - 2):
            raise ValueError(
                f'Argument fold (={fold}) must be between [0, ..., {n_sets-2}] as last '
                'fold is test set.')

        self.space_indices = {
            'train': np.concatenate([sets[f] for f in train_folds]),
            'valid': sets[fold],
            'test': sets[-1],
            'all': np.concatenate(sets_all)
        }

    def set_sparse_grid(
            self,
            gap_size: int,
            bounds=None) -> None:
        """Assign training and validation set based on a sparse grid.

        Every x-th x 2 pixel in longitude and latitude direction is used for training.

        Parameters
        ----------
        gap_size
            Integer > 0 indicating the grid distance between training and validation pixels.
        bounds
            The AOI bounds, an iterable of four numeric values: (min lat, max lat, min lon, max lon). If passed,
            the area is reduced to the specified extent. If this is not ``None``, no check is done if
            ``cv_mask_indices`` exist in mask.

        """

        mask = self.getxrvar('mask')
        sparse_grid = self.get_cv_grid(mask, gap_size)

        mask = (mask > 0).astype(int)
        mask *= sparse_grid
        mask = mask.values

        self.space_indices = {
            'train': np.argwhere(mask == 1),
            'valid': np.argwhere(mask == 2),
            'test': np.argwhere(mask == 2),
            'all': np.argwhere(np.isin(mask, [1, 2]))
        }

    def get_cv_grid(
            self,
            x,
            gap_size,
            latoffset=0,
            lonoffset=0,
            no_mask=False):
        """Create grid with distance of `gap_size` between pixels of two classes.

        Parameters
        ----------
        x
            xarray.Dataset or xarray.DataArray with `lat` and `lon` dimension.
        gap_size
            Integer > 0 indicating the grid distance between training and validation pixels.
        offset
            Integer > 0 indicating offset (from top left) to start from.
        no_mask
            If `True`, No masking is done.

        Returns
        ----------
        A xarray.DataArray, size as x.lat x x.lon with values 0=masked, 1=class1, 2=class2.

        """

        nlat = len(x.lat)
        nlon = len(x.lon)
        r = np.zeros((nlat, nlon), dtype=int)

        if no_mask:
            r[:] = 1
        else:
            current_start = 1
            for lat in np.arange(latoffset, nlat, gap_size):
                current_class = current_start
                for lon in np.arange(lonoffset, nlon, gap_size):
                    r[lat, lon] = current_class
                    current_class = current_class % 2 + 1
                current_start = current_start % 2 + 1

        m = xr.DataArray(r, coords=(x.lat, x.lon))

        return m

    def get_masks(self):
        """Returns maps of masks for train, valid and test set."""
        x = self.getxrvar('mask')
        ds = xr.Dataset()

        for cv_set in ['train', 'valid', 'test', 'all']:
            indices = self.space_indices[cv_set]
            mask = np.zeros((len(x.lat), len(x.lon)), dtype=int)
            for (lat, lon) in indices:
                mask[lat, lon] = 1
            ds[cv_set] = xr.DataArray(mask, coords=[x.lat, x.lon])

        return ds

    def set_rep_years(self, n: int, ref_var: str) -> None:
        """set how many years from the features get repeated.

        Sometimes it is necessary to extend the warmup period in order to get stable state
        variables. Here, you can set how many (random) years to repeat and which variable
        swerves as reference (Should be a feature, and all features should have same time dim).

        Parameters
        ----------
        n
            Number of years to sample.
        ref_var
            Reference variable to take time indices from.

        """

        self.rep_years_n = n
        self.feature_var = ref_var

    def get_sampler(self, sample_set: str) -> 'Bucket':
        """Get a sampler for train, validation or test set.

        Parameters
        ----------
        sample_set
            CV-set, one of 'train', 'valid', 'test'.

        Returns
        ----------
        A sampler (of type Bucket) that sampes from corresonding set.

        """

        if self.space_indices is None:
            raise ValueError(
                'You must set the mask IDs that correspond to the sampling sets using '
                '``Bucket.set_space_ids(...)`` first before calling Bucket.get_sampler(...).')

        if sample_set not in ['train', 'valid', 'test', 'all']:
            raise ValueError('Arg ``sample_set`` must be on of {``train``, ``valid``, ``test``, ``all``} but '
                             f'is ``{sample_set}``.')

        sampler = copy.copy(self)
        sampler._sample_set = sample_set
        ds = self.features_dynamic[0]
        time_slice = sampler.get_slice(ds)
        sampler._feature_time = sampler.getvartime(
            ds)[time_slice[0]:time_slice[1]]
        sampler._mean_std = {ds: sampler.get_mean_std(
            ds) for ds in sampler.datasets_list}
        sampler._datasets = sampler._sample_formatter.datasets

        return sampler

    def set_time_slices(self, ds: SpatialData) -> None:
        """Set time_slices defining train, validate and test time spans.

        The datasets contained in the bucket can cover different time-ranges and have varying
        temporal resolution. Sampling from the datacan be done for a 'train', 'valid' or 'test'
        set, each potentially covering a different time-range. Here, the needed information to read
        from the coorect time-range (and potentially also the time bins) are written to a given
        dataset 'ds', such that the information can be read while sampling.

        Note: Empty slice and bins encoded as None.

        Parameters
        ----------
        ds
            Dataset of type SpatialData to add to the bucket.

        """

        train_slice = self._sample_formatter['time_slices']['train']
        valid_slice = self._sample_formatter['time_slices']['valid']
        test_slice = self._sample_formatter['time_slices']['test']
        all_slice = self._sample_formatter['time_slices']['all']

        warmup = self._sample_formatter['warmup']

        if ds.hastime:
            train_slice, train_bins = ds.get_time_coverage(
                train_slice, warmup=warmup)
            valid_slice, valid_bins = ds.get_time_coverage(
                valid_slice, warmup=warmup)
            test_slice, test_bins = ds.get_time_coverage(
                test_slice, warmup=warmup)
            all_slice, all_bins = ds.get_time_coverage(
                all_slice, warmup=warmup)

            attrs_new = {
                'time_slices': {
                    'train': {
                        'slice': train_slice,
                        'bins': train_bins
                    },
                    'valid': {
                        'slice': valid_slice,
                        'bins': valid_bins
                    },
                    'test': {
                        'slice': test_slice,
                        'bins': test_bins
                    },
                    'all': {
                        'slice': all_slice,
                        'bins': all_bins
                    }
                }
            }
        else:
            attrs_new = {
                'time_slices': {
                    'train': {
                        'slice': None,
                        'bins': None
                    },
                    'valid': {
                        'slice': None,
                        'bins': None
                    },
                    'test': {
                        'slice': None,
                        'bins': None
                    },
                    'all': {
                        'slice': None,
                        'bins': None
                    }
                }
            }

        self.root[ds.varname_new].attrs.update(attrs_new)
        # self.getvar(ds.varname_new).attrs.update(attrs_new)

    def set_mask(self, mask: SpatialData) -> 'Bucket':
        """Add/replace Bucket mask.

        Parameters
        ----------
        mask
            A mask with dientions lat & lon of type SpatialData that indicates sampling set positions,
            e.g. unique values 0, 1, 2 represent: 0 = non-valid (convention). 1 & 2: used to switch sets
            by determining pixels to sample from.

        """
        if type(mask) is not SpatialData:
            raise ValueError(
                'Argument ``mask`` must be of type ``SpatialData``.')
        if not ((self.nlat == mask.nlat) & (self.nlon == mask.nlon)):
            raise ValueError(
                f'You specified bucket dimensions as nlat: {self.nlat}, nlon: {self.nlon} but the passed '
                f'mask has dimensions nlat: {mask.nlat}, nlon: {mask.nlon}.\nBucket path: {self._path}')
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(
                'Mask values must be of type int (numpy Int also allowed). You can either cast the datase '
                'or use mask = SpatialData(..., dtype=int).')

        # Force variable name.
        mask._varname_new = 'mask'

        self._add_data(mask, harmonize_dtype=False)

        return self

    def getvar(self, varname: str) -> zarr.core.Array:
        groups = list(self.root.group_keys())
        if varname not in groups:
            raise ValueError(
                f'Dataset ``{varname}``` not found in the Bucket, valid datasets are {groups}.')
        return self.root[varname]['data']

    def getxrvar(self, varname: str) -> xr.Dataset:
        groups = list(self.root.group_keys())
        if varname not in groups:
            raise ValueError(
                f'Dataset ``{varname}``` not found in the Bucket, valid datasets are {groups}.')
        path = os.path.join(self.path, varname)

        return xr.open_zarr(path)['data']

    def getvartime(self, varname) -> np.ndarray:
        groups = list(self.root.group_keys())
        if varname not in groups:
            raise ValueError(
                f'Dataset ``{varname}``` not found in the Bucket, valid datasets are {groups}.')
        return xr.open_zarr(os.path.join(self.path, varname)).time.data

    def add(self, *other: 'Bucket') -> 'Bucket':
        """Add one or multiple SpatialDatasets to the data bucket."""
        for element in other:
            _ = self.__add__(element)

        return self

    def read_data(self, lat: int, lon: int, standardize: str = 'features') -> Dict[str, Any]:
        """Read a sample (lat lon) from the bucket.

        Parameters
        ----------
        lat, lon
            Latitude, longitude index (!) to read from. standardize: Whether to standardize data, options
            are {'all', 'features', 'targets', 'none'}, default is 'features'.
        standardize
            Which data to standardize, one of `features` (default), `targets`, `all`, `none`.

        Returns
        ----------
        A dict following the structure defined in the SampleFormatter.
        Dynamic data has shape: seq_len x num_features
        Static data has shape: conv_size x conv_size x num_features

        """
        if self._sample_set is None:
            raise ValueError(
                'You cannot sample from this Bucket directly, use '
                'my_sampler = my_bucket.get_sampler(sample_set) to get Bucket you can sample from.'
            )
        if self._feature_time is None:
            raise ValueError(
                'You cannot sample from this Bucket directly, use'
                'my_sampler = my_bucket.get_sampler(sample_set) to get Bucket you can sample from.'
            )

        dataset_feat = {}
        dataset_targ = {}

        for ds_key, ds_value in self._datasets.items():
            hastime = ds_value['hastime']
            istarget = ds_value['istarget']

            ds = self.root[ds_key]

            m, s = self._mean_std[ds_key]

            standardize_this = False
            if standardize == 'all':
                standardize_this = True
            elif standardize == 'features':
                if not istarget:
                    standardize_this = True
            elif standardize == 'targets':
                if istarget:
                    standardize_this = True
            elif standardize == 'none':
                standardize_this = False
            else:
                raise ValueError(
                    'Argument ``standardize`` must be one of {"all", "features", "targets", "none"}.')

            try:
                if hastime:
                    t0, t1 = ds.attrs['time_slices'][self._sample_set]['slice']

                    if standardize_this:
                        data = (ds['data'][t0:t1, ..., lat, lon]
                                [..., np.newaxis] - m) / s
                    else:
                        data = ds['data'][t0:t1, ...,
                                          lat, lon][..., np.newaxis]

                else:
                    if standardize_this:
                        data = (ds['data'][..., lat, lon] - m) / s

                    else:
                        data = ds['data'][..., lat, lon]

            except Exception as e:
                message = '\nCannot read the dataset {} defined in the sample formatter '\
                    'config file: {}'.format(
                        ds_key, self._sample_formatter.cf_path)
                raise type(e)(str(e) + message)

            if istarget:
                dataset_targ.update({ds_key: data})
            else:
                dataset_feat.update({ds_key: data})

        return dataset_feat, dataset_targ

    def get_mean_std(self, ds: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get bins for a given dataset.

        Parameters
        ----------
        ds
            Dataset name, str.

        Returns
        ----------
        A 2-tuple of floats, the mean(s) and standard deviation(s) of the dataset.
        """

        attrs = self.root[ds].attrs
        return np.array(attrs['mean'], dtype=self._dtype), np.array(attrs['std'], dtype=self._dtype)

    def get_bins(self, ds: str) -> List:
        """Get bins for a given dataset.

        Parameters
        ----------
        ds: Dataset name, str.

        Returns
        ----------
        A list, the bins defining how a reference time range maps to the dataset's time dimentions .
        """
        if self._sample_set is None:
            raise ValueError(
                'You cannot get the bins for this bucket, use '
                'my_sampler = my_bucket.get_sampler(sample_set).')

        return self.root[ds].attrs['time_slices'][self._sample_set]['bins']

    def get_slice(self, ds: str) -> Iterable[int]:
        """Get bins for a given dataset.

        Parameters
        ----------
        ds
            Dataset name, str.

        Returns
        ----------
        A list of length two, the lower and upper bound to cut the dataset's time dimension to match
        a reference time range.
        """
        if self._sample_set is None:
            raise ValueError(
                'You cannot get the bins for this bucket, use '
                'my_sampler = my_bucket.get_sampler(sample_set).')
        return self.root[ds].attrs['time_slices'][self._sample_set]['slice']

    def _rm_empty_dirs(self) -> None:
        """Delete empty directories."""
        while True:
            to_delete = []
            for root, dirs, _ in os.walk(self._path):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if all(s.startswith('.') for s in os.listdir(full_path)):
                        to_delete.append(full_path)

            if to_delete:
                for p in to_delete:
                    shutil.rmtree(p)
            else:
                break

    def _check_path_exists(self, path: str) -> Tuple[str, bool]:
        """Check if path exists, return path and bool.

        Parameters
        ----------
        path
            The path to check.

        Returns
        ----------
        Returns a tuple
            - Path; same as the argument ``path``.
            - A boolean indicating if path exists or not.

        """
        if os.path.isdir(path):
            return path, True
        else:
            return path, False

    def _check_resolution(self, other: SpatialData) -> int:
        """Compare other resolution to this.

        Parameters
        ----------
        other
            A SpatialData object to be compared to this Bucket.

        Returns
        ----------
        A scale factor indicating how much higher the resolution of ``other`` is compared to this Bucket.

        """
        same_resolution = True
        if self._nlat != other.nlat:
            same_resolution = False
        if self._nlon != other.nlon:
            same_resolution = False

        if not same_resolution:
            # Infer scale factor.
            scale_lat = other.nlat / self._nlat
            scale_lon = other.nlon / self._nlon
            if scale_lat != scale_lon:
                raise ValueError(
                    'Scale factors for latitude and longitude must be equal:\n',
                    f'Scale lat: {scale_lat}, scale lon: {scale_lon}')
            if scale_lat % 1 != 0:
                raise ValueError(
                    'Scale factor must be integer, the dataset spatial dimensions must be a multiple '
                    f'of the Bucket reference resolution but is {scale_lat}.')
            if scale_lat < 1:
                raise ValueError(
                    'Scale factor must be 1 (same size as Bucket) or an integer larger than 1 (higher '
                    f'resolutin as Bucket) but is {scale_lat}.')
            return int(scale_lat)
        else:
            return 1

    def _get_array_paths(self, ds: SpatialData) -> str:
        """Get target path from dataset.

        Parameters
        ----------
        ds
            SpatialDataset to get target path for.

        Returns
        ----------
        A path.

        """
        return os.path.join(self._path, ds.varname_new)

    def _initialize(self) -> None:
        """Initialize metadata."""
        is_empty = len(self.attrs) == 0
        if is_empty:
            # There is no metadata yet, assert that required arguments are passed and then
            # initialize the metadata.
            for arg, arg_name in [[self._nlon, 'nlon'],
                                  [self._nlat, 'nlat'],
                                  [self._chunk_size, 'chunk_size']]:
                if arg is None:
                    raise ValueError(
                        f'You must pass the argument ``{arg_name}`` as you chose to overwrite the Bucket or '
                        f'it does not exist at specified location:\n{self.path}')
            self._attrs_update(OrderedDict({
                'nlat': self._nlat,
                'nlon': self._nlon,
                'chunk_size': self._chunk_size,
                'data': {}
            }))
        else:
            # The metadata does already exist. Assert that all required fields are present and
            # if the arguments - if passed - match the metadata.

            # Check metadata file - these fields are required:
            required_fields = [
                'nlat', 'nlon', 'chunk_size', 'data'
            ]
            for k in required_fields:
                assert k in self.attrs.keys(), 'Field \'{}\' not found in Bucket '\
                    'metadata, try to rebuild it.'.format(k)

            # Compare arguments to metadata.
            for arg, arg_name in [[self._nlon, 'nlon'],
                                  [self._nlat, 'nlat'],
                                  [self._chunk_size, 'chunk_size']]:
                # If the argument is empty, no need to check as it is retrieved from metadata.
                if arg is not None:
                    # Compare.
                    assert arg == self.attrs[arg_name], 'Argument \'{}\' ({}) does not match '\
                        'the (already existing) Bucket property \'{}\' ({}). You cannot change '\
                        '\'{}\' - either do not pass the argument and go with the specified '\
                        'value or overwrite the existing Bucket.'.format(
                            arg_name, arg, arg_name, self.attrs[arg_name], arg_name)

            # Retrieve attributes from metadata - this is redundant if passed arguments match
            # metadata values.
            self._nlat = self.attrs['nlat']
            self._nlon = self.attrs['nlon']
            self._chunk_size = self.attrs['chunk_size']

    def _json_encoder(self, obj: Any) -> Any:
        """Serializing fails when adding attribute of numpy type.

        Parameters
        ----------
        obj
            An object to serialize.

        Returns
        ----------
        The serialized object.

        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _set_mean_std(self, ds: SpatialData, scale_factor: int) -> None:
        """Add mean and std to Bucket array as attributes.

        Parameters
        ----------
        ds
            Dataset of type SpatialData.
        scale_factor
            Dataset scale factor.
        """

        mask = self.getvar('mask')[:] > 0

        # Expand mask to match ds lat x lon.
        mask = mask.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)
        preserve_rel = self._sample_formatter.datasets[ds.varname_new]['preserve_rel']

        # Get values where mask. If 'preserve_rel', mean and std are calculated for all variables,
        # else per varialbe (applies to static variables only if this is behavior for temporal
        # varialbes anyway).
        if ds.hastime or preserve_rel:
            # Expand dimensions to match ds dims.
            mask = np.broadcast_to(mask, (*ds.var.shape,))
            values = ds.var.data[np.where(mask)]
            mean = self._json_encoder(np.nanmean(values))
            std = self._json_encoder(np.nanstd(values))
        # If dataset has no time dimension, mean/std are derived per variable in first dim.
        elif (not ds.hastime) and ds.var.ndim == 3:
            mean = []
            std = []
            for i in range(ds.var.shape[0]):
                values = ds.var.data[i][np.where(mask)]
                mean.append(self._json_encoder(np.nanmean(values)))
                std.append(self._json_encoder(np.nanstd(values)))
        elif (not ds.hastime) and ds.var.ndim == 2:
            values = ds.var.data[np.where(mask)]
            mean = self._json_encoder(np.nanmean(values))
            std = self._json_encoder(np.nanstd(values))
        else:
            raise ValueError('Something went wrong while calculating mean and std of the dataset ``{}``. '
                             'The dataset must have 3 dimension if it has a time dimension, 2 or 3 if not. '
                             'It has {} dimensions.'.format(ds.varname, ds.var.ndim))

        attrs_new = {
            'mean': mean,
            'std': std
        }

        self.root[ds.varname_new].attrs.update(attrs_new)

    def _set_unit(self, ds: SpatialData) -> None:
        self.root[ds.varname_new].attrs.update({'unit': ds.unit})

    def get_unit(self, ds: str) -> str:
        return self.root[ds].attrs['unit']

    def _add_data(self, ds: SpatialData, harmonize_dtype: bool = True) -> 'Bucket':
        """Ad data to bucket.

        Parameters
        ----------
        ds
            Dataset of type SpatialData to add to the bucket.
        harmonize_dtype (default: True)
            Whether to use the Bucket's default dtype or keep the dtype of ds.

        Returns
        ----------
        The Bucket object.

        """
        if not isinstance(ds, SpatialData):
            raise ValueError(
                f'Argument is not of type ``SpatialData`` but {type(ds)}.')

        scale_factor = self._check_resolution(ds)

        if harmonize_dtype:
            dtype = self.dtype
        else:
            dtype = ds.dtype

        ds_write = ds.get_writeable(scale_factor)

        chunks = {'lat': self._chunk_size, 'lon': self._chunk_size}

        if ds.hastime:
            chunks.update({'time': -1})

        if ds._vardim is not None:
            chunks.update({'var': -1})

        if scale_factor > 1:
            chunks.update({'lat_index': -1, 'lon_index': -1})

        encoding = {
            'lat': {'compressor': None},
            'lon': {'compressor': None},
            'data': {'dtype': dtype, 'compressor': None}
        }

        ds_write = ds_write.chunk(chunks)

        ds_write.to_zarr(self.root.store, group=ds.varname_new,
                         mode='w', encoding=encoding)

        # Add global attributes to Bucket.
        attrs = {
            ds.varname_new: {
                'scale': scale_factor,
                'hastime': ds.hastime,
                'hastimebin': ds.hastimebin
            }
        }
        self._attrs_add_data(attrs)

        if ds.varname_new != 'mask':
            # Add time_slices info attributes to array.
            self.set_time_slices(ds)

            # Add mean and std attributes to array.
            self._set_mean_std(ds, scale_factor)

        # Add variable units attributes to array.
        self._set_unit(ds)

        return self

    def _attrs_init(self) -> None:
        """Initialize attributes of bucket."""
        self._attrs.update({
            'nlat': self.nlat, 'nlon': self.nlon, 'chunk_size': self.chunk_size, 'data': {}
        })

    def _attrs_update(self, data: Dict[str, SpatialData], *keys: Union[Iterable[str], str]) -> None:
        """Update yaml file (dict style).

        The keys are interpreted as hierarchical levels, i.e. if no element in keys, 'ds' is
        is added at first level, if keys passed, 'data' is added at the lower hierarchical levels.

        Parameter
        ---------
        ds
            Data to add to dict.
        *keys
            For sup-dicts, interpreted as hierarchical levels.

        """
        if not isinstance(data, dict):
            raise ValueError(
                'Argument ``data`` must be of type ``dict``.')

        attrs = self._attrs
        for k in keys:
            attrs = attrs[k]
        attrs.update(data)

        # This writes the changes to file.
        self._attrs.update(self.attrs)

    def _attrs_delete(self, *keys: Union[Iterable[str], str]) -> None:
        """Delete element.

        The keys are interpreted as hierarchical levels, i.e. if only one element
        in keys, the key is deleted from first level, if two from second etc.

        Parameters
        ----------
        *keys
            Keys to delete, multiple keys are interpreted as hierarchic levels.

        """
        nlevels = len(keys)
        attrs = self._attrs
        for l in range(nlevels - 1):
            attrs = attrs[keys[l]]
        del attrs[keys[-1]]

        # This writes the changes to file.
        self._attrs.update(self.attrs)

    def _attrs_add_data(self, ds: Dict[str, SpatialData]) -> None:
        """Add attributes for a given datasets to Bucket.

        Parameters
        ---------
        ds:
            Data to add to dict.
        """

        if not isinstance(ds, dict):
            raise ValueError(
                'Argument ``data`` must be of type ``dict``.')

        self._attrs_update(ds, 'data')

    def _attrs_delete_data(self, ds: SpatialData) -> None:
        """Deleta attributes for a given datasets to Bucket.

        Parameters
        ---------
        ds
            Data to add to dict.

        """

        if not isinstance(ds, SpatialData):
            raise ValueError(
                'Argument ``data`` must be of type ``SpatialData``.')
        if ds.varname_new not in self.attrs['data'].keys():
            raise ValueError(
                f'Dataset {ds.varname_new} not found in this Bucket.')
        self._attrs_delete('data', ds.varname_new)

    def __getitem__(self, idx: int) -> Tuple[Dict, Tuple[int, int]]:
        """Get a single sample from the Bucket.

        Parameters
        ----------
        idx
            The index of the sample to read.

        Returns
        ----------
        A tuple
            - A dictionary with the sample data.
            - A tuple with the sample latitude, longitude.

        """
        self.lat, self.lon = self.space_indices[self._sample_set][idx, :]
        feat, targ = self.read_data(self.lat, self.lon)
        return (feat, targ), (self.lat, self.lon)

    def __add__(self, other: SpatialData) -> 'Bucket':
        """Add a single SpatialDataset to the collection.

        Parameters
        ----------
        other
            The dataset to add to the Bucket.

        Returns
        ----------
        This bucket.

        """

        self._add_data(other)

        return self

    def __sub__(self, other: SpatialData) -> 'Bucket':
        """Remove a single SpatialDataset from the collection.

        Parameters
        ----------
        other
            The dataset to add to the Bucket.

        Returns
        ----------
        This bucket.

        """
        if not isinstance(other, SpatialData):
            raise ValueError(
                'Argument is not of type ``SpatialData`` but {type(other)}.'
            )

        rem_paths = self._get_array_paths(other)
        shutil.rmtree(rem_paths)
        self._rm_empty_dirs()

        # Add global attributes to Bucket.
        self._attrs_delete_data(other)

        return self

    def __call__(self) -> zarr.hierarchy.Group:
        return self.root

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        summary = ['Bucket @ {}'.format(
            self._path), '-' * (len(self._path) + 12)]
        for key in np.setdiff1d(list(self.attrs.keys()), 'data'):
            summary.append('{}: {}'.format(key, self.attrs[key]))
        summary.append('data:')
        for key in list(self.attrs['data'].keys()):
            data_info = '  - {}:{}scale:{:3d}, hastime: {:5s}, hastimebin: {:5s}'.format(
                key, ' ' * (10 - len(key)), self.attrs['data'][key]['scale'],
                str(self.attrs['data'][key]['hastime']),
                str(self.attrs['data'][key]['hastimebin']))
            summary.append(data_info)

        return u'\n'.join(summary)

    def __len__(self) -> int:
        if self.space_indices is None:
            raise ValueError(
                'The ``space_indices`` have not been set yet, use the method '
                '``self.set_sample_sets(...)`` to do so.')
        n = self.space_indices[self._sample_set].shape[0]
        return n

    def shape(self, key: str) -> int:
        return self.root[key]['data'].shape

    @property
    def nlat(self) -> int:
        return self._nlat

    @property
    def nlon(self) -> int:
        return self._nlon

    @property
    def path(self) -> str:
        return self._path

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def attrs(self) -> dict:
        """Dictionary of global attributes on this dataset."""
        return self.root.attrs.asdict()

    @property
    def _attrs(self) -> zarr.attrs.Attributes:
        """Global attributes of this dataset."""
        return self.root.attrs

    @property
    def sample_formatter(self) -> SampleFormatter:
        return self._sample_formatter

    @property
    def features(self) -> Dict:
        return self._sample_formatter.features

    @property
    def features_dynamic(self) -> Dict:
        return self._sample_formatter.features_dynamic

    @property
    def features_static(self) -> Dict:
        return self._sample_formatter.features_static

    @property
    def targets(self) -> Dict:
        return self._sample_formatter.targets

    @property
    def num_features(self) -> int:
        return len(self._sample_formatter.num_features)

    @property
    def num_features_dynamic(self) -> int:
        return len(self._sample_formatter.num_featues_dynamic)

    @property
    def num_features_static(self) -> int:
        return len(self._sample_formatter.num_features_static)

    @property
    def num_targets(self) -> int:
        return len(self._sample_formatter.num_targets)

    @property
    def datasets_list(self) -> List[str]:
        return self.features + self.targets


_use_shared_memory = False

np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class BucketCollator(object):
    """Custom collate function to generate batches.

    Parameters
    ----------
    feature_time
        Time dimension of features.
    n_rep_years
        Number of years to repeat, will be randomly sampled from non-leap years.

    """

    def __init__(self, sampler: Bucket, n_rep_years: int):

        self.time = sampler._feature_time
        self.use_msc_spinup = sampler.use_msc_spinup
        self.n_rep_years = n_rep_years

        self.dynamic_features = sampler.features_dynamic

        self.years = np.unique(self.time.astype(
            'datetime64[Y]')).astype(str)
        # Remove leap years bc this leads to different length batches.
        # self.years = np.setdiff1d(years, np.arange(1972, 2050, 4).astype(str))

    def __call__(self, batch):
        """Puts each data field into a tensor with outer dimension batch size"""

        (feat, targ), locations = self.collate(batch)

        # Select random years from features and put them into a sequence
        # for model spin-up.
        if not self.use_msc_spinup:
            indices = self.get_random_year_ind()

            feat_spinup = {}

            for k, v in feat.items():
                if k in self.dynamic_features:
                    feat_spinup[k] = v[:, indices, ...]

        # Calculate the mean seasonal cycle from features and put them
        # into a sequence for model spin-up.
        else:
            indices = self.get_year_indices()
            n_years = len(indices)

            feat_spinup = {}

            for k, v in feat.items():
                if k in self.dynamic_features:
                    msc = torch.zeros_like(v[:, :365, ...])
                    tile_reps = [1] * msc.ndim
                    tile_reps[1] = n_years
                    for start_idx, end_idx in indices:
                        msc += v[:, start_idx:end_idx, ...]
                    msc /= n_years
                    feat_spinup[k] = msc.repeat(*tile_reps)

        return feat_spinup, feat, targ, locations

    def get_random_year_ind(self) -> np.array:
        """Get indices for a random years for train, validation or test set, if leap years cut last day.

        The indices correspond to the daily time dimention, such that full years can be extracted from
        daily time series.

        Returns
        ----------
        Indices (np.array) corresponding time dimension indices.

        """

        if (self.n_rep_years is None) or (self.n_rep_years == 0):
            indices = np.arange(len(self.time))

        else:
            indices = np.empty(0, dtype=int)
            for i in range(self.n_rep_years):
                random_year = np.random.choice(self.years)
                start_date = np.datetime64('{}-01-01'.format(random_year))
                end_date = np.datetime64('{}-12-31'.format(random_year))

                start_idx = np.argwhere(self.time == start_date)[0][0]
                end_idx = np.argwhere(self.time == end_date)[0][0] + 1

                # We discard last day of leap years to have uniform length time-series.
                if end_idx - start_idx == 366:
                    end_idx -= 1

                if end_idx - start_idx != 365:
                    raise ValueError(
                        'The selected indices to compute the mean seasonal cycle for '
                        f'the spinup result in a perode of length {end_idx - start_idx} '
                        f'for the year {random_year}, expected length 365.'
                    )

                indices = np.concatenate(
                    (indices, np.arange(start_idx, end_idx)))

        return indices

    def get_year_indices(self) -> np.array:
        """Get start / end indices of years, if leap years cut last day.

        The indices correspond to the daily time dimention, such that full years are extracted from
        daily time series.

        Returns
        ----------
        Indices (np.array) corresponding time dimension indices.

        """

        indices = []
        if (self.n_rep_years is None) or (self.n_rep_years == 0):
            pass

        else:
            for year in self.years:
                start_date = np.datetime64('{}-01-01'.format(year))
                end_date = np.datetime64('{}-12-31'.format(year))

                start_idx = np.argwhere(self.time == start_date)[0][0]
                end_idx = np.argwhere(self.time == end_date)[0][0] + 1

                # We discard last day of leap years to compute the mean seasonal cycle
                # based on 365 day year.
                if end_idx - start_idx == 366:
                    end_idx -= 1

                if end_idx - start_idx != 365:
                    raise ValueError(
                        'The selected indices to compute the mean seasonal cycle for '
                        f'the spinup result in a perode of length {end_idx - start_idx} '
                        f'for the year {year}, expected length 365.'
                    )
                indices.append((start_idx, end_idx))

        return indices

    def collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(error_msg_fmt.format(elem.dtype))

                return self.collate([torch.from_numpy(b) for b in batch])
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(batch[0], torch._six.int_classes):
            return torch.tensor(batch)
        elif isinstance(batch[0], torch._six.string_classes):
            return batch
        elif isinstance(batch[0], torch._six.container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
            return type(batch[0])(* (torch._utils.collate.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(batch[0], torch._six.container_abcs.Sequence):
            transposed = zip(*batch)
            return [self.collate(samples) for samples in transposed]

        raise TypeError((error_msg_fmt.format(type(batch[0]))))


def getDataloader(
        dataset: Bucket,
        cv_set: str,
        batch_size: int = 32,
        seed: int = None,
        nworkers: int = 0,
        n_rep_years: int = 0,
        pin_memory=True,
        **kwargs) -> Bucket:
    """Returns a dataloader object to load data samples.

    Parameters
    ----------
    datasetcd
        Data bucket object that implements data loading functionalities.
    cv_set
        Cross-validation set, one of {'train', 'valid', 'test'}.
    batch_size (default: 32)
        The minibatch size.
    seed (default: None)
        The random seed. The seed will be changed + worker id if nworkers is > 0. Thus, results can
        only be reproduced with same seed and same nworkers.
    nworkers (default: 0)
        The number of subprocesses to use for data loading. 0 means
        that the data will be loaded in the main process.
    n_rep_years:
        Number of years spin-up years.
    **kwargs
        Passed to the DataLoader (torch.utils.data.DataLoader).

    """

    def worker_init_fn(worker_id: int) -> Callable:
        """Initialize data loader workers, used to set seed for samplers.

        Parameters
        ----------
        worker_id
            The worker id.

        Returns
        ----------
        init_fn
            The initialization function.
        """

        if seed is not None:
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            # torch.random.manual_seed(worker_seed)
            torch.random.manual_seed(seed)

        return

    if cv_set == 'train':

        sampler = dataset.get_sampler('train')

        dataloader = torch.utils.data.DataLoader(
            sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nworkers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=BucketCollator(sampler, n_rep_years)
        )

    elif cv_set == 'valid':

        sampler = dataset.get_sampler('valid')

        dataloader = torch.utils.data.DataLoader(
            sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nworkers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=BucketCollator(sampler, n_rep_years)
        )

    elif cv_set == 'test':

        sampler = dataset.get_sampler('test')

        dataloader = torch.utils.data.DataLoader(
            sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nworkers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=BucketCollator(sampler, n_rep_years)
        )

    elif cv_set == 'valid_plot':

        sampler = dataset.get_sampler('valid')

        sampler.space_indices = dict(
            train=np.array([0]),
            valid=np.vstack(
                (sampler.space_indices['valid'][40], sampler.space_indices['valid'][100])),
            test=np.array([0])
        )

        dataloader = torch.utils.data.DataLoader(
            sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nworkers,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=BucketCollator(sampler, n_rep_years)
        )

    elif cv_set == 'all':

        sampler = dataset.get_sampler('all')

        dataloader = torch.utils.data.DataLoader(
            sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nworkers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=BucketCollator(sampler, n_rep_years)
        )

    else:
        raise ValueError(
            f'Invalid argument ``cv_set``: {cv_set}. Must be one of "train", "valid", "test".')

    return dataloader


def check_bucket(bucket: Bucket, tasks: Iterable[str]) -> None:
    """Consistency check for data bucket.

    Parameters
    ----------
    bucket
        The data bucket to check.
    tasks
        The training arguments.

    """

    bucket.getvartime('tair')

    available_tasks = set(bucket.targets)
    for task in tasks:
        if task not in available_tasks:
            raise ValueError(
                f'Task {task} not available, possible choices: {tasks}.')

    features = bucket.features_dynamic
    for i in range(1, len(features)):
        if not np.all(bucket.getvartime(features[0]) == bucket.getvartime(features[i])):
            raise ValueError('{Time dimension of all features must be equal.')
