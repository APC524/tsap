 # do inference for various time series models

import numpy as np
from scipy import stats
import scipy

def acovf(x, demean=True, fft=False):
    '''
    Autocovariance for 1D time series array
	Input:
    x : array
        Time series data. Must be 1d.
    demean : bool
        If True, then subtract the mean x from each element of x
    fft : bool
        If True, use FFT convolution.  This method should be preferred
        for long time series.
	Output:
    acovf : array
        autocovariance function
    '''
	# the input might be a n by one matrix
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)
    n = len(x)

    if demean:
        xc = x - x.mean()
    else:
        xc = x

	# denominator
    a = np.arange(1, n + 1)
    d = np.hstack((a, a[:-1][::-1]))

    if fft:
        nobs = len(xc)
        Frf = np.fft.fft(xc, n=nobs * 2)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[n - 1:]
        return acov.real
    else:
        return (np.correlate(xc, xc, 'full') / d)[n - 1:]


def acf(x, nlags = 40, demean=True, fft=False):
	"""
	Compute the autocorrelation coefficients

	which is defined by \rho_k = cov_k / cov_0,
	where \cov_k is the k-th order autocovariance

	The input is the same as acovf function, the output is an array of autocorrelation coefficients
	"""
	cov_array = acovf(x, demean, fft)
	return ( cov_array[:nlags+1] /cov_array[0])

def BL_stat(acf_array, nobs):
    """
    Compute Box-Ljung test statistic

	Input:
    acf_array : an array of acf_functions
    nobs : int
        Number of observations in the entire sample

  	Output:
    -------
    q-stat : array
        Ljung-Box Q-statistic for autocorrelation parameters
    p-value : array
        P-value of the Q statistic
	result:
    Notes
    ------
    Written to be used with acf.
    """


    if(np.abs( acf_array[0] - 1) < 1e-10):
        acf_array = acf_array[1:]  #remove lag0


	test_stat = (nobs * (nobs + 2) ) * np.cumsum( (1. / (nobs - np.arange(1, len(acf_array) + 1))) * acf_array**2)

    chi2 = stats.chi2.sf(test_stat, np.arange(1, len(acf_array) + 1))

    test_result = chi2 > 0.05

    return test_stat, chi2, test_result


def yule_walker(x, order = 1,  demean=True, method = 'unbaised'):
    """
    Input:
    x: 1d array, time series
    order: int, order of the AR models
    demean: bool, whether subtract the mean or Notes
    method: "unbiased" or "mle"

    Output:
    rho: 1d array of length order, coefficients of the model
    sigma: variance of X_t
    """
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)
    n = len(x)

    if demean:
        x = x - x.mean()


    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError("ACF estimation method must be 'unbiased' or 'MLE'")

    if method == "unbiased":        # this is df_resid ie., n - p
        denom = lambda k: n - k
    else:
        denom = lambda k: n
    if x.ndim > 1:
        raise ValueError("the first input should be a vector")
    r = np.zeros(order+1, np.float64) # first order covariance
    r[0] = (x**2).sum() / denom(0)
    for k in range(1,order+1):
        r[k] = (x[0:-k]*x[k:]).sum() / denom(k)
    R = scipy.linalg.toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigma = r[0] - (r[1:]*rho).sum()

    return rho, sigma




def arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c',
                         model_kw={}, fit_kw={}):
    """
    Returns information criteria for many ARMA models

    Parameters
    ----------
    y : array-like
        Time-series data
    max_ar : int
        Maximum number of AR lags to use. Default 4.
    max_ma : int
        Maximum number of MA lags to use. Default 2.
    ic : str, list
        Information criteria to report. Either a single string or a list
        of different criteria is possible.
    trend : str
        The trend to use when fitting the ARMA models.
    model_kw : dict
        Keyword arguments to be passed to the ``ARMA`` model
    fit_kw : dict
        Keyword arguments to be passed to ``ARMA.fit``.

    Returns
    -------
    obj : Results object
        Each ic is an attribute with a DataFrame for the results. The AR order
        used is the row index. The ma order used is the column index. The
        minimum orders are available as ``ic_min_order``.

    Examples
    --------

    >>> from statsmodels.tsa.arima_process import arma_generate_sample
    >>> import statsmodels.api as sm
    >>> import numpy as np

    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> arparams = np.r_[1, -arparams]
    >>> maparam = np.r_[1, maparams]
    >>> nobs = 250
    >>> np.random.seed(2014)
    >>> y = arma_generate_sample(arparams, maparams, nobs)
    >>> res = sm.tsa.arma_order_select_ic(y, ic=['aic', 'bic'], trend='nc')
    >>> res.aic_min_order
    >>> res.bic_min_order

    Notes
    -----
    This method can be used to tentatively identify the order of an ARMA
    process, provided that the time series is stationary and invertible. This
    function computes the full exact MLE estimate of each model and can be,
    therefore a little slow. An implementation using approximate estimates
    will be provided in the future. In the meantime, consider passing
    {method : 'css'} to fit_kw.
    """
    from pandas import DataFrame

    ar_range = lrange(0, max_ar + 1)
    ma_range = lrange(0, max_ma + 1)
    if isinstance(ic, string_types):
        ic = [ic]
    elif not isinstance(ic, (list, tuple)):
        raise ValueError("Need a list or a tuple for ic if not a string.")

    results = np.zeros((len(ic), max_ar + 1, max_ma + 1))

    for ar in ar_range:
        for ma in ma_range:
            if ar == 0 and ma == 0 and trend == 'nc':
                results[:, ar, ma] = np.nan
                continue

            mod = _safe_arma_fit(y, (ar, ma), model_kw, trend, fit_kw)
            if mod is None:
                results[:, ar, ma] = np.nan
                continue

            for i, criteria in enumerate(ic):
                results[i, ar, ma] = getattr(mod, criteria)

    dfs = [DataFrame(res, columns=ma_range, index=ar_range) for res in results]

    res = dict(zip(ic, dfs))

    # add the minimums to the results dict
    min_res = {}
    for i, result in iteritems(res):
        mins = np.where(result.min().min() == result)
        min_res.update({i + '_min_order' : (mins[0][0], mins[1][0])})
    res.update(min_res)

    return Bunch(**res)
