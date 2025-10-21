from cpython cimport array as c_array
from cython.parallel import prange, parallel, threadid
from array import array
cimport cython
cimport libc.math
cimport libc.limits
from scipy.stats import beta as betadist


@cython.cdivision(True)
cdef double digamma(double x) nogil:
    """
    Compute the digamma function.
    Implementation adapted from that of Bernardo (1976).
    """

    cdef double s = 1e-5
    cdef double c = 8.5
    cdef double s3 = 8.333333333e-2
    cdef double s4 = 8.333333333e-3
    cdef double s5 = 3.968253968e-3
    cdef double d1 = -0.5772156649

    cdef double r
    cdef double y
    cdef double v

    if x > s:
        y = x
        v = 0.0

        while y < c:
            v -= 1.0 / y
            y += 1.0

        r = 1.0 / y
        v += libc.math.log(y) - r / 2.0
        r = 1.0 / (y * y)
        v -= r * (s3 - r * (s4 - r * s5))
    else:
        v = d1 - 1.0 / x

    return v


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fit_alpha_beta(object impressions_arr, object clicks_arr, double alpha0=1.5, double beta0=5,int niter=10000):
    """
    Fit betabinomials coefficients
    :param impressions_arr:
    :param clicks_arr:
    :param alpha0:
    :param beta0:
    :param niter:
    :return:
    """

    cdef double[:] impressions = array('d',impressions_arr)
    cdef double[:] clicks = array('d',clicks_arr)

    cdef double alpha_old=alpha0
    cdef double beta_old=beta0
    cdef double alpha
    cdef double beta
    cdef int it, jj,N
    N=len(clicks)

    cdef double numerator,denominator,c,i
    for it in xrange(niter):

        numerator=0
        denominator=0
        for jj in prange(N,nogil=True):
            c=clicks[jj]
            i=impressions[jj]
            numerator += digamma(c + alpha_old) - digamma(alpha_old)

            denominator+=digamma(i + alpha_old+beta_old) - digamma(alpha_old+beta_old)

        alpha=alpha_old*numerator/denominator

        numerator=0
        denominator=0
        for jj in prange(N,nogil=True):
            c=clicks[jj]
            i=impressions[jj]
            numerator += digamma(i-c + beta_old) - digamma(beta_old)

            denominator+=digamma(i + alpha_old+beta_old) - digamma(alpha_old+beta_old)


        beta=beta_old*numerator/denominator

        #print 'alpha {} | {}  beta {} | {}'.format(alpha,alpha_old,beta,beta_old)

        if abs(alpha-alpha_old)<1e-10 and abs(beta-beta_old)<1e-10:
            #print 'early stop'
            break

        alpha_old=alpha
        beta_old=beta

    return alpha,beta


#The posterior distribution for rates | clicks ~ Beta(clicks + alpha, imps - clicks + beta)
#hence we can compute all statistical properties such as mean, variance and ppf

def get_rates_posterior_rv(imps,clicks,alpha_hat,beta_hat):

    return betadist(clicks+alpha_hat,imps-clicks+beta_hat)

def posterior_mean(imps,clicks, alpha_hat,beta_hat):
  return (clicks + alpha_hat ) / (imps + alpha_hat + beta_hat)

def posterior_interval_bounds(x,imps,clicks,alpha_hat,beta_hat):
    """
    Endpoints of the range that contains value percent of the distribution
    :param x: float between 0 and 1
    :return:
    """
    rv=get_rates_posterior_rv(imps,clicks,alpha_hat,beta_hat)
    return rv.interval(x)


def posterior_cdf(x,imps,clicks,alpha_hat,beta_hat):
    """
    P(rate <=value)
    :param value: float between 0 and 1
    :return:
    """
    rv=get_rates_posterior_rv(imps,clicks,alpha_hat,beta_hat)
    return rv.cdf(x)

def posterior_sf(x,imps,clicks,alpha_hat,beta_hat):
    """
    P(rate >value)
    :param value: float between 0 and 1
    :return:
    """
    rv=get_rates_posterior_rv(imps,clicks,alpha_hat,beta_hat)
    return rv.sf(x)


# def fit_alpha_beta_py(impressions, clicks, alpha0=1.5, beta0=5,niter=1000):
#
#
#     alpha_old=alpha0
#     beta_old=beta0
#
#     for it in range(niter):
#
#         alpha=alpha_old*\
#         (sum(psi(c + alpha_old) - psi(alpha_old) for c,i in zip(clicks,impressions)))/\
#         (sum(psi(i + alpha_old+beta_old) - psi(alpha_old+beta_old) for c,i in zip(clicks,impressions)))
#
#
#         beta=beta_old*\
#         (sum(psi(i-c + beta_old) - psi(beta_old) for c,i in zip(clicks,impressions)))/\
#         (sum(psi(i + alpha_old+beta_old) - psi(alpha_old+beta_old) for c,i in zip(clicks,impressions)))
#
#
#         #print 'alpha {} | {}  beta {} | {}'.format(alpha,alpha_old,beta,beta_old)
#         sys.stdout.flush()
#
#         if np.abs(alpha-alpha_old) and np.abs(beta-beta_old)<1e-10:
#             #print 'early stop'
#             break
#
#         alpha_old=alpha
#         beta_old=beta
#
#     return alpha,beta