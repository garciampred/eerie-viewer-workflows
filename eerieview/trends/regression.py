import numba
import numpy
from numba_stats import t


@numba.njit(nogil=True, parallel=False, cache=True)
def mklr(x: numpy.ndarray, y: numpy.ndarray) -> tuple:
    """
    Perform univariate linear regression of y on x.

    Parameters
    ----------
    x : numpy.ndarray
        Column vector of predictors. Must be of same length as y.
    y : numpy.ndarray
        Column vector of observations. Must be of same length as x.

    Returns
    -------
    list
        A list containing the following regression statistics:
        - r0 : float
            Regression line intercept
        - sig0 : float
            Standard error in r0
        - r1 : float
            Regression line slope
        - sig1 : float
            Standard error in r1

    Raises
    ------
    NotImplementedError
        If the length of x and y do not match.

    Notes
    -----
    This function performs univariate linear regression of y on x:
              y = r0 + r1*x
    and returns estimates for r0, r1, and their standard errors sig0 and sig1
    respectively.

    This implementation is coded from the Dover's book by A.A.Sveshnikov
    "Problems in Probability Theory,...", p.326, --AK, Feb 27  2004, and
    was adapted to R by the Santander Met. Group in Nov 2020.

    The error estimates for the regression coefficients are calculated using
    the formulae given in Sveshnikov's book and assume that the errors in y
    are normally distributed and independent with constant variance.

    Oct 11, 2011: This description has been added, no changes in code
    """
    if len(x) != len(y):
        raise NotImplementedError("Length of vectors does not match.")

    n = len(x)
    s0 = n
    s1 = numpy.nansum(x)
    s2 = numpy.nansum(numpy.square(x))
    v0 = numpy.nansum(y)
    v1 = numpy.nansum(numpy.multiply(x, y))

    r0 = (s2 * v0 - s1 * v1) / (s2 * s0 - s1**2)
    r1 = (-s1 * v0 + s0 * v1) / (s2 * s0 - s1**2)

    e = y - (r0 + numpy.multiply(r1, x))
    Smin = numpy.nansum(e**2)

    sig0 = numpy.sqrt(s2 / (s2 * s0 - s1**2) * Smin / (n - 2))
    sig1 = numpy.sqrt(s0 / (s2 * s0 - s1**2) * Smin / (n - 2))

    return (r0, sig0, r1, sig1)


@numba.njit(cache=True, parallel=False, nogil=True)
def ltr_OLSdofrNaN(x, y, p=0.90) -> tuple:
    """
    Compute linear trend slopes, their confidence intervals and some other related stats.

    It is used for timeseries using Santer et al. (2008) method modified for dealing
    with missing data in time series.

    Please note:
    (1) the time grid is expected to be uniform, and
    (2) missing data values contain NaN
    (3) if this function is called with 2 input parameters only, p is assumed
        to be NaN, and NaN is returned for cinthw; other output parameters
        are unaffected
    (4) this function uses an external function mklr.m for the OLS regression

    Parameters
    ----------
    x : array_like
        A 1-D array of time values (a uniform grid).
    y : array_like
        A 1-D array of data values (NaN in place of missing values).
    p : float, optional
        A confidence level for the uncertainty interval (0 < p < 1).

    Returns
    -------
    b : float
        The estimated slope of the linear trend.
    cinthw : float or NaN
        The half-width of the confidence interval.
    sig : float
        The estimated standard error in b.
    DOFr : float
        The reduced number of DOF (a.k.a. effective sample size).
    rho : float
        The lag-1 autocorrelation coefficient of data residuals (w.r.t. trend line).
    pval : float
        The p-value of the estimated b in the two-sided Student's t test for the null
        hypothesis of no trend.
    irrc : int
        The "irregularity" code:
            irrc = 0, regular application of the algorithm
            irrc = 1, rho < 0 (rho=0 value is used in sig and cinthw calculations,
                        but unmodified rho value is returned)
            irrc = 10, DOFr < 3; in this case results of the calculation are not
                        recommended for use; when DOFr-->2-0, values of sig and cinthw
                        tend to infinity and pval-->1 (Inf are returned for sig and
                        cinthw and 1 for pval when DOFr <= 2)
            irrc = 100, rho could not be estimated (NaN are returned for rho, Inf for
                        sig and cinthw, and 1 for pval)
            irrc = 1000, no calculation is done b/c number of available data
                        points Na < 3
    N : int
        The timeseries length.
    a : float
        The intercept of the trend line.
    Na : int
        The number of available data points.
    Nc : int
        The sample size for calculating rho.

    References
    ----------
    Santer, B. D., Thorne, P. W., Haimberger, L., Taylor, K. E.,
    Wigley, T. M. L., Lanzante, J. R., Solomon, S., Free, M., Gleckler, P. J.,
    Jones, P. D., Karl, T. R., Klein, S. A., Mears, C., Nychka, D.,
    Schmidt, G. A., Sherwood, S. C. and Wentz, F. J. (2008), Consistency of
    modelled and observed temperature trends in the tropical troposphere.
    International Journal of Climatology, 28: 1703-1722, doi:10.1002/joc.1756.

    """
    ## Step 0
    irrc = 0
    b = numpy.nan
    sig = numpy.inf
    pval = 1.0
    cinthw = numpy.inf
    DOFr = 0
    rho = numpy.nan
    Nc = numpy.nan
    a = numpy.nan

    N = len(x)
    # x and y are assumed to be vectors of the same length and here are enforced
    # both to be columns (of the size Nx1):
    # x = reshape(x,N,1);
    # y = reshape(y,N,1);

    ia = numpy.nonzero(y)[0]
    # ia = ia.reshape((len(ia), 1))
    Na = len(ia)

    if Na < 3:
        irrc += 1000
    else:
        ya = y[ia]
        xa = x[ia]

        # Step 1
        # calling the regular OLS (mklr() is an external function)
        # for available data
        a, sa, b, sb = mklr(xa, ya)

        # Step 2
        # computing data residuals w.r.t. the OLS trend line:
        ea = ya - a - b * xa

        e = numpy.zeros_like(x).astype("float64")
        e[:] = numpy.nan
        e[ia] = ea

        # Step 3
        # lag-1 autocorrelation cefficient

        EE = numpy.zeros((N - 1, 2))
        EE[:] = numpy.nan
        EE[:, 0] = e[: N - 1]
        EE[:, 1] = e[1:N]

        ic = numpy.argwhere(~numpy.isnan(EE.sum(axis=1))).ravel()
        Nc = len(ic)
        if Nc < 2:
            irrc = irrc + 100
        else:
            Ec = EE[ic, :]
            rho = numpy.corrcoef(Ec[:, 0], Ec[:, 1])[0, 1]

        if rho != rho:
            irrc += 100
        else:
            # Step 4
            rhop = max(rho, 0)
            if rho < 0:
                irrc += 1

            # Step 5
            # Computing reduced number of DOF
            DOFr = Na * (1 - rhop) / (1 + rhop)  # type: ignore
            # Step 6
            if DOFr < 3:
                irrc += 10

            # Step 7
            if DOFr > 2:
                # Adjust estimated standard error in the trend slope
                sig = sb * numpy.sqrt((Na - 2) / (DOFr - 2))
                pval = (
                    2
                    * (
                        1
                        - t.cdf(
                            numpy.array([numpy.absolute(b) / sig]),
                            DOFr - 2,
                            loc=0.0,
                            scale=1.0,
                        )
                    )[0]
                )
                # half-width of the (p*100)% confidence interval for b:
                cinthw = (
                    sig
                    * t.ppf(numpy.array([0.5 + p / 2]), DOFr - 2, loc=0.0, scale=1.0)[0]
                )

    return b, cinthw, sig, DOFr, rho, pval, irrc, N, a, Na, Nc
