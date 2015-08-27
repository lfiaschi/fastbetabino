from cpython cimport array as c_array
from cython.parallel import prange, parallel, threadid
from array import array
cimport cython
from scipy.stats import beta as betadist
from scipy.stats import kstest
import numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt

cdef extern from "gsl/gsl_sf_gamma.h":
      double  gamma "gsl_sf_gamma" (double x) nogil
      double  lgamma "gsl_sf_lngamma" (double x) nogil

cdef extern from "gsl/gsl_sf_psi.h":
      double  digamma "gsl_sf_psi" (double x) nogil

cpdef log_likelihood(object impressions_arr, object clicks_arr, double alpha=1.0,  double beta=1.0):
    """
    Log likelihood for the betabinomial model
    :param impressions_arr:
    :param clicks_arr:
    :param alpha:
    :param beta:
    :return:
    """

    cdef double[:] impressions = array('d',impressions_arr)
    cdef double[:] clicks = array('d',clicks_arr)

    cdef int N,it

    cdef double res, positive, negative, LGAB, LGA, LGB, c, i

    N = len(clicks)

    res=0.0

    LGAB = lgamma(alpha+beta)
    LGA = lgamma(alpha)
    LGB = lgamma(beta)

    for it in xrange(N):
        c = clicks[it]
        i = impressions[it]
        positive = LGAB + lgamma(c+alpha) + lgamma(i-c+beta)
        negative = lgamma(i+alpha+beta) + LGA + LGB

        res+=positive-negative

    return res

cdef double dalpha_log_likelihood(double i ,double c, double alpha=1.0,  double beta=1.0) nogil:
    """
    derivative respect to beta of the log likelihood for the betabinomial model
    :param i:
    :param c:
    :param alpha:
    :param beta:
    :return:
    """

    cdef double res, DAB, DA, DB

    DAB = digamma(alpha+beta)
    DA = digamma(alpha)
    DB = digamma(beta)

    res = DAB-digamma(i+alpha+beta)+digamma(c+ alpha)-DA
    return res

cdef double dbeta_log_likelihood(double i ,double c, double alpha=1.0,  double beta=1.0) nogil:
    """
    derivative respect to beta of the log likelihood for the betabinomial model
    :param i:
    :param c:
    :param alpha:
    :param beta:
    :return:
    """

    cdef double res, DAB, DA, DB

    DAB = digamma(alpha+beta)
    DA = digamma(alpha)
    DB = digamma(beta)

    res = DAB-digamma(i+alpha+beta)+digamma(i-c+ beta)-DB
    return res

from scipy.optimize import minimize
def fit_alpha_beta_lbfgs(impressions_arr, clicks_arr, alpha0=1.0, beta0=1.0 ):

    func = lambda x: - log_likelihood(impressions_arr,clicks_arr,x[0],x[1])

    opt = minimize(func,(alpha0,beta0),bounds=[(0.0001,1000),(0.0001,1000)],
                   tol=1e-10,method='L-BFGS-B', options={'maxiter':1000})
    
    return opt['x']



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fit_alpha_beta_sgd(object impressions_arr, object clicks_arr,
                   double alpha0=1.0, double beta0=1.0,
                   double eta0=1e-2,
                   int nepochs=100,
                   int num_threads=1,
                   int batch_size= 1000,
                   float tol=1e-10
                   ):
    """
    Fit betabinomials coefficients
    :param impressions_arr:
    :param clicks_arr:
    :param alpha0:
    :param beta0:
    :param nepochs:
    :param num_threads number of threads for the summations
    :return:
    """

    assert len(impressions_arr) == len(clicks_arr), 'clicks {}!={} impressions'.format(len(impressions_arr),len(clicks_arr))
    cdef double[:] impressions = array('d',impressions_arr)
    cdef double[:] clicks = array('d',clicks_arr)

    cdef double alpha_old=alpha0
    cdef double beta_old=beta0
    cdef double alpha
    cdef double beta
    cdef size_t it, jj,N, ARLEN, start_read,stop_read, t

    N = batch_size
    ARLEN=len(clicks)
    alpha=alpha_old
    beta=beta_old
    cdef double dalpha,dbeta,c,i
    t=0
    for it in xrange(nepochs):
        #shuffle_data(impressions,clicks,ARLEN)
        start_read = 0
        stop_read = N
        t+=1
        #print 'it = {}'.format(it)
        while stop_read<=ARLEN:

            dalpha=0
            dbeta=0
            for jj in xrange(start_read,stop_read):
                c=clicks[jj]
                i=impressions[jj]
                dalpha += dalpha_log_likelihood(i,c,alpha_old,beta_old)
                dbeta += dbeta_log_likelihood(i,c,alpha_old,beta_old)


            # dalpha,dbeta = num_der(impressions_arr,clicks_arr,alpha_old,beta_old)


            eta = eta0
            alpha=alpha_old + eta*dalpha
            beta=beta_old + eta*dbeta
            alpha= max(0.000001,alpha)
            beta = max(0.000001,beta)
            alpha = min(alpha,1000)
            beta = min(beta,1000)

            #print dalpha,num_der(impressions_arr,clicks_arr,alpha_old,beta_old)[0]
            print alpha,beta,dalpha,dbeta, log_likelihood(impressions_arr,clicks_arr,alpha,beta),stop_read


            if abs(alpha-alpha_old) <tol and abs(beta-beta_old)<tol:
                break

            alpha_old=alpha
            beta_old=beta

            start_read+=N
            stop_read+=N

    return alpha,beta



def alpha_beta_from_mean_variance(impressions_arr, clicks_arr):

    ctr_arr = clicks_arr/(impressions_arr+1e-72)

    mu = np.mean(ctr_arr)
    sigma2 = np.var(ctr_arr)

    alpha = ( (1-mu)/sigma2 - 1/mu ) * mu**2
    beta = alpha *( 1 /mu - 1)

    return alpha, beta

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fit_alpha_beta(object impressions_arr, object clicks_arr,
                   double alpha0=1.0, double beta0=1.0,
                   int niter=1000,
                   int num_threads=1,
                   float tol=1e-10):
    """
    Fit betabinomials coefficients
    :param impressions_arr:
    :param clicks_arr:
    :param alpha0:
    :param beta0:
    :param niter:
    :param num_threads number of threads for the summations
    :param tol: tolerance for the stopping criterion
    :return:
    """

    cdef double[:] impressions = array('d',impressions_arr)
    cdef double[:] clicks = array('d',clicks_arr)

    cdef double alpha_old=alpha0
    cdef double beta_old=beta0
    cdef double alpha
    cdef double beta
    cdef int it, jj,N

    N = len(clicks)

    cdef double numerator,denominator,c,i
    for it in xrange(niter):

        numerator=N*(- digamma(alpha_old))
        denominator=N*(- digamma(alpha_old+beta_old))
        for jj in prange(N,nogil=True,num_threads=num_threads):
            c=clicks[jj]
            i=impressions[jj]
            numerator += digamma(c + alpha_old) #- digamma(alpha_old)

            denominator+=digamma(i + alpha_old+beta_old) #- digamma(alpha_old+beta_old)

        alpha=alpha_old*numerator/denominator

        numerator=N*(- digamma(beta_old))
        denominator=N*( - digamma(alpha_old+beta_old) )
        for jj in prange(N,nogil=True,num_threads=num_threads):
            c=clicks[jj]
            i=impressions[jj]
            numerator += digamma(i-c + beta_old) #- digamma(beta_old)

            denominator+=digamma(i + alpha_old+beta_old) # - digamma(alpha_old+beta_old)


        beta=beta_old*numerator/denominator


        #print alpha,beta, - log_likelihood(impressions_arr,clicks_arr,alpha,beta)

        #print 'alpha {} | {}  beta {} | {}'.format(alpha,alpha_old,beta,beta_old)

        if abs(alpha-alpha_old) <tol and abs(beta-beta_old)<tol:
            #print 'early stop'
            break

        alpha_old=alpha
        beta_old=beta

    return alpha,beta



cdef shuffle_data(double[:] impressions_arr, double[:] clicks_arr,size_t n):
    """
    Utility function to shuffle the data between minibatches
    :param impressions_arr:
    :param clicks_arr:
    :param n:
    :return:
    """
    cdef size_t i,j;
    cdef double t;


    for i in range(n):
        j = i + rand() / (RAND_MAX / (n - i) + 1)
        t = impressions_arr[j]
        impressions_arr[j] = impressions_arr[i]
        impressions_arr[i] = t

        t = clicks_arr[j]
        clicks_arr[j] = clicks_arr[i]
        clicks_arr[i] = t



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fit_alpha_beta_minibatch(object impressions_arr, object clicks_arr,
                   double alpha0=1.0, double beta0=1.0,
                   int niter=10000,
                   int num_threads=1,
                   int batch_size= 1000,
                   float tol=1e-10
                   ):
    """
    Fit betabinomials coefficients
    :param impressions_arr:
    :param clicks_arr:
    :param alpha0:
    :param beta0:
    :param niter:
    :param num_threads number of threads for the summations
    :return:
    """

    assert len(impressions_arr) == len(clicks_arr), 'clicks {}!={} impressions'.format(len(impressions_arr),len(clicks_arr))
    cdef double[:] impressions = array('d',impressions_arr)
    cdef double[:] clicks = array('d',clicks_arr)

    cdef double alpha_old=alpha0
    cdef double beta_old=beta0
    cdef double alpha
    cdef double beta
    cdef size_t it, jj,N, ARLEN, start_read,stop_read

    N = batch_size
    ARLEN=len(clicks)

    cdef double numerator,denominator,c,i
    # shuffle_data(impressions,clicks,ARLEN)

    for it in xrange(niter):
        shuffle_data(impressions,clicks,ARLEN)
        start_read = 0
        stop_read = N

        while (stop_read<=ARLEN):
            numerator=(stop_read -  start_read)*(- digamma(alpha_old))
            denominator=(stop_read -  start_read)*(- digamma(alpha_old+beta_old))
            for jj in prange(start_read,stop_read,nogil=True,num_threads=num_threads):
                c=clicks[jj]
                i=impressions[jj]
                #TODO: this can be further optimized by moving this digamma (alpha_old) from the loop
                numerator += digamma(c + alpha_old) #- digamma(alpha_old)

                denominator += digamma(i + alpha_old+beta_old) #- digamma(alpha_old+beta_old)

            alpha=alpha_old*numerator/denominator

            numerator=(stop_read -  start_read)*(- digamma(beta_old))
            denominator=(stop_read -  start_read)*(- digamma(alpha_old+beta_old))
            for jj in prange(start_read,stop_read,nogil=True,num_threads=num_threads):
                c=clicks[jj]
                i=impressions[jj]
                numerator += digamma(i-c + beta_old) # - digamma(beta_old)

                denominator += digamma(i + alpha_old+beta_old)# - digamma(alpha_old+beta_old)


            beta=beta_old*numerator/denominator

            #print 'alpha {} | {}  beta {} | {}'.format(alpha,alpha_old,beta,beta_old)

            if abs(alpha-alpha_old) <tol and abs(beta-beta_old)<tol:
                break

            alpha_old=alpha
            beta_old=beta

            start_read+=N
            stop_read+=N

            # if stop_read>=ARLEN:
            #     #print 'reshuffling batch {}  start = {} stop = {} '.format(N,start_read,stop_read)
            #     shuffle_data(impressions,clicks,ARLEN)
            #     start_read=0
            #     stop_read=N

    return alpha,beta

from math import gamma as gammafunc
from cmath import log


# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def log_likelihood(object impressions_arr, object clicks_arr,
#                    double alpha=1.0, double beta=1.0):
#     """
#     Fit betabinomials coefficients
#     :param impressions_arr:
#     :param clicks_arr:
#     :param alpha0:
#     :param beta0:
#     :param niter:
#     :param num_threads number of threads for the summations
#     :param tol: tolerance for the stopping criterion
#     :return:
#     """
#
#     cdef double[:] impressions = array('d',impressions_arr)
#     cdef double[:] clicks = array('d',clicks_arr)
#
#     cdef double alpha_tilde
#     cdef double beta_tilde
#     cdef int it
#     cdef double i,c,res
#
#     N = len(clicks)
#
#     for it in xrange(N):
#         i = impressions[it]
#         c = clicks[it]
#         res+=lgamma(alpha+beta)  + lgamma(c+alpha) + lgamma(i-c+beta) -  \
#     (lgamma(i+alpha+beta) + lgamma(alpha) + lgamma(beta))
#
#
#
#
#     return res


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

def goodness_fit_rates(imps,clicks,alpha_hat,beta_hat, confidence=0.05):
    """
    Estimates whether the fit can be rejected to a certain confidence level
    :param imps:
    :param clicks:
    :param alpha_hat:
    :param beta_hat:
    :return:
    """

    rates = [c/(float(i) + 1e-72) for i,c in zip(imps,clicks)]

    rv = betadist(alpha_hat, beta_hat)
    D,p = kstest(rates,rv.cdf)
    print rates
    return D, p , p<confidence

