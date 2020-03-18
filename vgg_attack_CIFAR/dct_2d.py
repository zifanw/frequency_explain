import numpy as np
from scipy.fftpack import dct

def dct_2d(x):
    return dct(dct(x.T).T)