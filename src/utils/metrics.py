"""
Calculate metrics like correlation or rmse on multidimensional array along given dimentsions
using dask.

Metrics implemented:
 * correlation           > xr_corr
 * rmse                  > xr_rmse
 * mean percentage error > xr_mpe
 * bias                  > xr_bias
 * phaseerr              > xr_phaseerr
 * varerr                > xr_varerr
 * modeling effficiency  > xr_mef

Only values present in both datasets are used to calculate metrics.

"""

import numpy as np
import xarray as xr
import warnings
from datetime import datetime


def pearson_cor_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)
        valid_count = valid_values.sum(axis=dims)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        mod_ -= np.nanmean(mod_, axis=dims, keepdims=True)
        obs_ -= np.nanmean(obs_, axis=dims, keepdims=True)

        cov = np.nansum(mod_ * obs_, axis=dims) / valid_count
        std_xy = (np.nanstd(mod_, axis=dims) * np.nanstd(obs_, axis=dims))

        corr = cov / std_xy

        return corr


def xr_corr(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        pearson_cor_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'corr', 'units': '-'})
    if isinstance(mod, xr.DataArray):
        m.name = 'corr'
    return m


def mse_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        se = np.power(mod-obs, 2)
        mse = np.nanmean(se, axis=dims)

        return mse

def xr_mse(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        mse_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'MSE'})
    if isinstance(mod, xr.DataArray):
        m.name = 'mse'
    return m

def xr_rmse(mod, obs, dim):
    m = np.sqrt(xr_mse(mod, obs, dim))

    m.attrs.update({'long_name': 'RMSE'})
    if isinstance(mod, xr.DataArray):
        m.name = 'rmse'
    return m


def mpe_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mpe = 100 * np.nanmean((obs - mod) / obs, axis=dims)

        return mpe


def xr_mpe(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        mpe_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'mpe', 'units': '%'})
    if isinstance(mod, xr.DataArray):
        m.name = 'mpe'
    return m

def stdratio_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)
        valid_count = valid_values.sum(axis=dims)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        return np.nanstd(mod_, dims) / np.nanstd(obs_, dims)

def xr_stdratio(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        stdratio_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'std ratio'})
    if isinstance(mod, xr.DataArray):
        m.name = 'stdratio'
    return m

def bias_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        return np.nanmean(mod_, axis=dims) - np.nanmean(obs_, axis=dims)


def xr_bias(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        bias_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'bias'})
    if isinstance(mod, xr.DataArray):
        m.name = 'bias'
    return m

def xr_bias2(mod, obs, dim):
    m = xr_bias(mod, obs, dim) ** 2
    m.attrs.update({'long_name': 'bias2'})
    if isinstance(mod, xr.DataArray):
        m.name = 'bias2'
    return m


def varerr_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        return np.square(np.nanstd(mod_, dims) - np.nanstd(obs_, dims))


def xr_varerr(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        varerr_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'varerr'})
    if isinstance(mod, xr.DataArray):
        m.name = 'varerr'
    return m


def phaseerr_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        return (1.0 - pearson_cor_gufunc(mod_, obs_, dims)) * 2.0 * np.nanstd(mod_, dims) * np.nanstd(obs_, dims)


def xr_phaseerr(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        phaseerr_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'phaseerr'})
    if isinstance(mod, xr.DataArray):
        m.name = 'phaseerr'
    return m


def rel_bias_gufunc(mod, obs, dims):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        return (np.nanmean(mod_, axis=dims) - np.nanmean(obs_, axis=dims)) / np.nanmean(obs_, axis=dims)


def xr_rel_bias(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        rel_bias_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'relative bias'})
    if isinstance(mod, xr.DataArray):
        m.name = 'rel_bias'
    return m


def mef_gufunc(mod, obs, dims):
    # x is obs, y is mod
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        sse = np.nansum(np.power(mod_-obs_, 2), axis=dims)
        sso = np.nansum(
            np.power(obs_-np.nanmean(obs_, axis=dims, keepdims=True), 2), axis=dims)

        mef = 1.0 - sse / sso

        return mef

def xr_mef(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        mef_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'NSE', 'units': '-'})
    if isinstance(mod, xr.DataArray):
        m.name = 'nse'
    return m

def mefrob_gufunc(mod, obs, dims):
    # x is obs, y is mod
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mod_ = mod.copy()
        obs_ = obs.copy()

        valid_values = np.isfinite(mod_) & np.isfinite(obs_)

        mod_[~valid_values] = np.nan
        obs_[~valid_values] = np.nan

        sse = np.power(np.nanmedian(np.abs(mod_-obs_), axis=dims), 2)
        sso = np.power(np.nanmedian(np.abs(obs_-np.nanmean(obs_, axis=dims, keepdims=True)), axis=dims), 2)

        mef = 1.0 - sse / sso

        return mef

def xr_mefrob(mod, obs, dim):
    dims = tuple([-i for i in range(len(dim), 0, -1)])
    m = xr.apply_ufunc(
        mefrob_gufunc, mod, obs, dims,
        input_core_dims=[dim, dim, []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'robust NSE', 'units': '-'})
    if isinstance(mod, xr.DataArray):
        m.name = 'nse_rob'
    return m

def get_metric(mod, obs, fun, dim='time', verbose=False):
    """Calculate a metric along a dimension.

    Metrics implemented:           name
    * correlation                  > corr
    * mse                          > mse
    * rmse                         > rmse
    * mean percentage error        > mpe
    * standard deviation ratio     > stdratio
    * bias                         > bias
    * bias squared                 > bias2
    * phaseerr                     > phaseerr
    * varerr                       > varerr
    * robust modeling effficiency  > nse_rob
    * modeling effficiency         > nse

    Only values present in both datasets are used to calculate metrics.

    Parameters
    ----------
    data: xarray.Dataset
        Dataset with data variables 'mod' (modelled) and 'obs' (observed).
    fun: Callable
        A function that takes three arguments: Modelled (xarray.DataArray), observed (xarray.DataArray)
        and the dimension along which the metric is calculated.
    dim: str
        The dimension name along which the metri is calculated, default is `time`.

    Returns
    ----------
    xarray.Dataset

    """

    return fun(mod, obs, dim)


def get_metrics(mod, obs, funs, dim='time', iter_dim=None, verbose=True):
    """Calculate multiple metrics along a dimension and combine into single dataset.

    Metrics implemented:           name
    * correlation                  > corr
    * mse                          > mse
    * rmse                         > rmse
    * mean percentage error        > mpe
    * standard deviation ratio     > stdratio
    * bias                         > bias
    * bias squared                 > bias2
    * phaseerr                     > phaseerr
    * varerr                       > varerr
    * robust modeling effficiency  > nse_rob
    * modeling effficiency         > nse


    Only values present in both datasets are used to calculate metrics.

    Parameters
    ----------
    mod: xarray.DataArray
        The modelled data.
    obs: xarray.DataArray
        The observed data.
    funs: Iterable[str]
        An iterable of function names (see `metrics implemented`).
    dim: str
        The dimension name along which the metri is calculated.
    iter_dim: str
        Optional. If passed, the given dimension of `mod` is iterated and
        the metrics are computed for each coordinate.
    verbose: bool
        Silent if False (True is default).

    Returns
    ----------
    xarray.Dataset

    """

    fun_lookup = {
        'corr': xr_corr,
        'mse': xr_mse,
        'rmse': xr_rmse,
        'mpe': xr_mpe,
        'stdratio': xr_stdratio,
        'bias': xr_bias,
        'bias2': xr_bias2,
        'rel_bias': xr_rel_bias,
        'nse': xr_mef,
        'nse_rob': xr_mefrob,
        'varerr': xr_varerr,
        'phaseerr': xr_phaseerr
    }

    requested_str = ", ".join(funs)
    options_str = ", ".join(fun_lookup.keys())

    if isinstance(dim, str):
        dim = [dim]

    tic = datetime.now()

    if verbose:
        print(f'{timestr(datetime.now())}: calculating metrics [{requested_str}]')

    met_list = []
    if iter_dim is None:
        for fun_str in funs:
            if verbose:
                print(f'{timestr(datetime.now())}: - {fun_str}')
            if fun_str not in fun_lookup:
                raise ValueError(
                    f'Function `{fun_str}` not one of the implemented function: [{options_str}].'
                )
            fun = fun_lookup[fun_str]
            met_list.append(fun(mod, obs, dim).compute())

    else:
        if iter_dim not in mod.dims:
            raise ValueError(f'`iter_dim` {iter_dim} not found in `mod`.')
        for fun_str in funs:
            iter_list = []
            for i in range(len(mod[iter_dim])):
                if verbose:
                    print(f'{timestr(datetime.now())}: - {fun_str}')
                if fun_str not in fun_lookup:
                    raise ValueError(
                        f'Function `{fun_str}` not one of the implemented function: [{options_str}].'
                    )
                fun = fun_lookup[fun_str]
                iter_list.append(fun(mod.isel({iter_dim: i}), obs, dim).compute())
            met_list.append(xr.concat(iter_list, dim='rep'))

    met = xr.merge(met_list)

    toc = datetime.now()

    elapsed = toc - tic
    elapsed_mins = int(elapsed.seconds / 60)
    elapsed_secs = int(elapsed.seconds - 60 * elapsed_mins)

    if verbose:
        print(f'{timestr(datetime.now())}: done; elapsed time: {elapsed_mins} min {elapsed_secs} sec')

    return met


def timestr(t):
    return t.strftime("%m/%d/%Y, %H:%M:%S")


def _single_xr_quantile(x, q, dim):
    if isinstance(dim, str):
        dim = [dim]
    ndims = len(dim)
    axes = tuple(np.arange(ndims)-ndims)
    m = xr.apply_ufunc(
        np.nanquantile, x,
        input_core_dims=[dim],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True,
        kwargs={'q': q, 'axis': axes})
    m.attrs.update({'long_name': f'{x.attrs.get("long_name", "")} quantile'})
    if isinstance(m, xr.DataArray):
        m.name = 'quantile'
    return m


def xr_quantile(x, q, dim):
    if not hasattr(q, '__iter__'):
        q = [q]
    quantiles = []
    for i, q_ in enumerate(q):
        r = _single_xr_quantile(x, q_, dim).compute()
        quantiles.append(r)
    quantiles = xr.concat(quantiles, 'quantile')
    quantiles['quantile'] = q

    return quantiles


def global_cell_size(n_lat=180, n_lon=360, normalize=False, radius=6371.0, return_xarray=True):
    """Grid size per lon-lat cell on a sphere.
    
    Surface area (km^2) per grid cell for a longitude-latitude grid on a
    sphere with given radius. The grid is defined by the number of cells
    in latitude (n_lat) and longitude (l_lon). If normalize is True, the
    values get divided by the total area, such that the values can be
    directly used for weighting.
    
    Args:
        n_lat: int
            Size of latitude dimension.
        n_lon: int
            Size of latitude dimension.
        normalize: Bool (default: False)
            If True, the values get notmalized by the maximum area.
        radius: Numeric
            The radius of the sphere, default is 6371.0 for average Earth radius.
        return_xarray: Bool
            Wheter to return an xarray.DataArray or a numpy array. If True, the lat/lon
            coordinates are derived from n_lat / n_lon arguments, check code for details.
    Returns:
        2D numpy array or xrray.Dataset of floats and shape n_lat x n_lon, unit is km^2.
    
    """
    lon = np.linspace(-180., 180, n_lon+1)*np.pi/180
    lat = np.linspace(-90., 90, n_lat+1)*np.pi/180
    r = radius
    A = np.zeros((n_lat, n_lon))
    for lat_i in range(n_lat):
        for lon_i in range(n_lon):
            lat0 = lat[lat_i]
            lat1 = lat[lat_i+1]
            lon0 = lon[lon_i]
            lon1 = lon[lon_i+1]
            A[lat_i, lon_i] = (r**2.
                               * np.abs(np.sin(lat0)
                                         - np.sin(lat1))
                               * np.abs(lon0
                                         - lon1))
            
    gridweights = A / np.sum(A) if normalize else A
    if return_xarray:
        gridweights = xr.DataArray(gridweights,
                                   coords=[np.linspace(90, -90, n_lat*2+1)[1::2], np.linspace(-180, 180, n_lon*2+1)[1::2]],
                                   dims=['lat', 'lon'])
    return gridweights

def weighted_avg_and_std(xdata, weights=None):
    """
    Return the weighted average and standard deviation.

    Args:
        xdata : xr.DataArray
        weights : xr.DataArray, same shape as xdata

    Returns:
        (weighted_mean, weighted_std)
    """
    
    assert isinstance(xdata, xr.DataArray), 'xdata must be xr.DataArray'
    if weights is None:
        weights = xr.ones_like(xdata)
    assert isinstance(weights, xr.DataArray), 'weights must be xr.DataArray'
    assert xdata.shape == weights.shape, 'shape of xdata and weights must be equal'

    xdata = xdata.data
    weights = weights.data
    
    weights = weights[np.isfinite(xdata)]
    xdata = xdata[np.isfinite(xdata)]

    assert np.all(np.isfinite(weights)), 'Some weight are missing where xdata is not nan'

    if weights is None:
        weights = np.ones_like(xdata)
    if np.all(np.isnan(xdata)):
        average = np.nan
        variance = np.nan
    else:
        average = np.average(xdata, weights=weights)
        # Fast and numerically precise:
        variance = np.average((xdata-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def transform_nse(x):
    return x.where(x>0, np.tanh(x))


def quantile_1d(data, weights, quant):
    """
    Compute the weighted quantile of a 1D numpy array.
    from https://github.com/nudomarinero/wquantiles/blob/master/weighted.py

    Parameters
    ----------
        data: nd-array for which to calculate mean and variance
        weights: nd-array with weights for data (same shape of data)
        quant: quantile to compute, it must have a value between 0 and 1.
    
    Returns
    -------
        The quantile.
    """

    mask = np.isfinite(data)

    data = data[mask]
    weights = weights[mask]
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quant > 1.) or (quant < 0.):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    sn = np.cumsum(sorted_weights)
    assert np.sum(sn) > 0, "The sum of the weights must not be zero!"
    pn = (sn - 0.5 * sorted_weights) / np.sum(sorted_weights)
    # Get the value of the weighted median
    # noinspection PyTypeChecker
    return np.interp(quant, pn, sorted_data)

def weighted_quantile(x, q):
    '''Calculate weighted spatial quantile.

    Parameters
    ----------
        x: an xarray dataset or array, must have dimensions `lat`and `lon`.
        q: the quantile, a number in in the interval [0 to 1].

    Returns
    -------
        The quantiles, same type as input without dimensions `lat` and `lon`.
     '''
    weights = np.cos(np.deg2rad(x.lat))
    weights, _ = xr.broadcast(weights, x.lon)
    m = xr.apply_ufunc(
        quantile_1d, x, weights, 0.5,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon'], []],
        dask='parallelized',
        output_dtypes=[float],
        vectorize=True,
        keep_attrs=True)
    return m

def weighted_median(x):
    '''Calculate weighted spatial median.

    Parameters
    ----------
        x: an xarray dataset or array, must have dimensions `lat`and `lon`.
        q: the quantile, a number in in the interval [0 to 1].

    Returns
    -------
        The median, same type as input without dimensions `lat` and `lon`.
     '''
    return weighted_quantile(x, 0.5)

def weighted_mean(x):
    '''Calculate weighted spatial mean.

    Parameters
    ----------
        x: an xarray dataset or array, must have dimensions `lat`and `lon`.

    Returns
    -------
        The mean, same type as input without dimensions `lat` and `lon`.
     '''
    weights = np.cos(np.deg2rad(x.lat))
    weights.name = "weights"
    x_weighted = x.weighted(weights)
    return x_weighted.mean(('lat', 'lon'))

def _calc_detrend(y):
    """ufunc to be used by linear_trend"""
    x = np.arange(len(y))

    mask = np.isfinite(y)

    if np.sum(mask) < 2:
        r = np.zeros_like(y)
        r[:] = np.nan
        return np.array([np.nan, np.nan])

    slope, intercept = np.polyfit(x[mask], y[mask], 1)

    return np.array([intercept, slope])

def detrend(obj, multiplicative=False):
    obj_y = obj.resample(time='1Y').mean()

    time = obj.time.dt
    s = time.dayofyear - 1
    year_fraction = np.where(time.is_leap_year, s / 366, s / 365)

    time_x = time.year - time.year.min() + year_fraction - 0.5

    r = xr.apply_ufunc(
        _calc_detrend, obj_y,
        vectorize=True,
        input_core_dims=[['time']],
        output_core_dims=[['coef']],
        output_dtypes=[np.float],
        dask_gufunc_kwargs={'output_sizes': {"coef": 2}},
        dask='parallelized')

    #fit = r.sel(coef=0) + r.sel(coef=1) * time_x
    fit = r.sel(coef=0) + r.sel(coef=1) * time_x

    if multiplicative:
        dt = obj / fit * obj_y.mean('time')
    else:
        dt = obj - fit

    return dt

def add_decomp(ds, do_detrend=False, multiplicative=False):
    '''Add MSC and IAV (trend removed).

    Parameters
    ----------
    ds: xarray dataset to add decompositions to.
    do_detrend: whether to detrend for calculation of MSC and IAV.
    multiplicative: whether to detrend multiplicative (preserve absolute zero). Can
        be a bool or a list of variable names for thich to use `multiplicative`. Default
        is False.
    '''

    for var in ds.data_vars:
        if isinstance(multiplicative, bool):
            mpt = multiplicative
        else:
            mpt = var in multiplicative
        if do_detrend:
            d = detrend(ds[var], multiplicative=mpt)
        else:
            d = ds[var]

        msc = d.groupby('time.month').mean('time').compute()
        ds[var + '_msc'] = msc
        ds[var + '_iav'] = (d.groupby('time.month') - msc).compute()


@xr.register_dataset_accessor("geo")
@xr.register_dataarray_accessor("geo")
class WeightedAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def weighted_mean(self):
        """Returns the cell-size weighted spatial mean."""
            
        return weighted_mean(self._obj)

    def weighted_median(self):
        """Returns the cell-size weighted spatial median."""
            
        return weighted_median(self._obj)

    def weighted_quantile(self, q):
        """Returns the cell-size weighted spatial quantile (q)."""
            
        return weighted_quantile(self._obj, q)

    def add_decomp(self, do_detrend=False, multiplicative=False):
        """Adds MSC and IAV (trend removed).

        Parameters
        ----------
        ds: xarray dataset to add decompositions to.
        do_detrend: whether to detrend for calculation of MSC and IAV.
        multiplicative: whether to detrend multiplicative (preserve absolute zero). Can be a bool
            or a list of variable names for thich to use `multiplicative`. Only relevant if 
            `detrend` is TrueDefault is False.
        """
        add_decomp(self._obj, do_detrend=do_detrend, multiplicative=multiplicative)
