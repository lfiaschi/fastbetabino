from fastbetabino import *
from array import array
import numpy as np


N=100000

imps = np.random.randint(1,N,N)
clicks = list()
for i in imps:
    clicks.append(np.random.randint(0,i))


from contextlib import contextmanager
import time
@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))

with timeit_context('test '):
    print fit_alpha_beta(imps,clicks)

