from fastbetabino import log_likelihood
from scipy.special import gamma as gammafunc
from math import log
from scipy.optimize import minimize
import numpy as np
alpha=20
beta=.5
N = 10000

imps = np.random.randint(1,N,N)
rates = np.random.beta(alpha,beta,N)
clicks = list()
for r,i in zip(rates,imps):
    clicks.append(np.random.binomial(i,r))

def fit_alpha_beta_lbfgs(impressions_arr, clicks_arr, alpha0=1.0, beta0=1.0 ):

    func = lambda x: - log_likelihood(impressions_arr,clicks_arr,x[0],x[1])

    def call(x):
        print -func(x),x

    opt = minimize(func,(alpha0,beta0),tol=1e-30,method='L-BFGS-B',
                   options={'maxiter':30,'disp':True}, bounds=[(0.0001,100),(0.0001,100)], callback=call)
    print opt
    return opt['x']

fit_alpha_beta_lbfgs(imps,clicks)