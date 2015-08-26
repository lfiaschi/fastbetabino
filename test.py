from fastbetabino import *
from array import array
import numpy as np



N=10000

imps = np.random.randint(1,N,N)
imps = np.array([float(el) for el in imps])
clicks = list()
for i in imps:
    clicks.append(float(np.random.randint(0,i)))

clicks = np.array([float(el) for el in clicks])
imps = imps[:5]
clicks = clicks[:5]

print [el for el in zip(imps,clicks)]
shuffle_data(imps,clicks, len(clicks))
print [el for el in zip(imps,clicks)]


raise
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


