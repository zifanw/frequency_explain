import numpy as np
import scipy
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
from load_cifar import load_cifar_10_data


def DCT(image):
    return dct(dct(image, norm="ortho", axis=0), norm="ortho", axis=1)


def iDCT(image):
    return idct(idct(image, norm="ortho", axis=0), norm="ortho", axis=1)


def FFT(image, spectrum=True):
    ftimage = np.fft.fft2(image, axes=(0, 1))
    ftimage = np.fft.fftshift(ftimage, axes=(0, 1))
    if spectrum:
        ftimage = np.abs(ftimage)
    return ftimage


def iFFT(ftimage):
    ftimage = ftimage = np.fft.ifftshift(ftimage, axes=(0, 1))
    image = np.abs(np.fft.fft2(ftimage, axes=(0, 1)))
    image /= np.max(image, axis=(0, 1))
    return image


def masking(image, mode='n', stride=1):
    dctSize = image.shape[0]
    dof = []
    dof_image = []
    for i in range(dctSize):
        freq_image = DCT(image.copy())
        if i > stride - 1:
            if mode == 'n':
                mask = np.ones_like(freq_image)
                mask[i - stride:i, :i] = 0
                mask[:i, i - stride:i] = 0
            elif mode == 's':
                mask = np.zeros_like(freq_image)
                mask[i - stride:i, :i] = 1
                mask[:i, i - stride:i] = 1
            freq_image *= mask
        dof.append(freq_image)
        dof_image.append(iDCT(freq_image) / 255.)
    return np.array(dof), np.array(dof_image)
