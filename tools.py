from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import random
import matplotlib.pyplot as plt


def cconv(a, b):
    '''
    Computes the circular convolution of the (real-valued) vectors a and b.
    '''
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def cconv2(a, b):
    '''
    Computes the circular convolution of the (real-valued) matrices a and b.
    '''
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b)).real

def gaussian(d2, sigma, A):
    '''
    computes A * exp(-d2/(2 * sigma**2))
    '''
    return A * np.exp(-d2/(2.0 * sigma**2))

def gaussian(center, sigma, A, size):
    if(len(size) == 1):
        return A * np.exp(-(np.arange(size[0]) - center)**2 / (2.0 * sigma**2))

def circular_gaussian(z_src, z_center, A, s, size):
    if(len(size) == 1):
        # We do it in 1D, return a column vector
        ''' z_src is a vector of size N
            z_center is a 1D tuple z_center=(x0,)'''
        N = size[0]
        d = np.min(np.array([np.fabs(z_src-z_center[0]), N - np.fabs(z_src-z_center[0])]), axis=0)
        return A * np.exp(-d**2/(2.0 * s**2))
    elif(len(size) == 2):
        ''' z_src is Nx2 dimensional vector 
        z_src = [i, j], i being line number, j column number 
        z_center is a 1x2 dimensional vector '''
        height, width = size
        d_i = np.min(np.array([np.fabs(z_src[:,0]-z_center[0]), height - np.fabs(z_src[:,0]-z_center[0])]), axis=0)
        d_j = np.min(np.array([np.fabs(z_src[:,1]-z_center[1]), width  - np.fabs(z_src[:,1]-z_center[1])]), axis=0)
        return np.reshape(A * np.exp(-(d_i**2 + d_j**2)/(2.0 * s**2)), size)
    else:
        print("Cannot compute the circular gaussian in more than 2 dimensions")

def plot_dnf_on_scenar(field, scenar,filename=None):
    Ihistories = []
    fuhistories = []
    field.reset()
    while(not scenar.is_finished()):
        field.step(scenar.get_input())
        Ihistories.append(scenar.get_input())
        scenar.step()
        fuhistories.append(field.get_output())
    Ihistories = np.array(Ihistories)
    fuhistories = np.array(fuhistories)
    
    fig = plt.figure()

    ax = fig.add_subplot(121)
    plt.imshow(Ihistories, cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
    plt.colorbar()
    ax.set_aspect('auto')

    ax = fig.add_subplot(122)
    plt.imshow(fuhistories, cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
    plt.colorbar()
    ax.set_aspect('auto')

    if(filename):
        plt.savefig(filename, bbox_inches='tight')


def circular_rectified_cosine(z_src, z_center, A, s, size):
    if(len(size) == 1):
        # We do it in 1D, return a column vector
        ''' z_src is a vector of size N
            z_center is a 1D tuple z_center=(x0,)'''
        N = size[0]
        d = np.min(np.array([np.fabs(z_src-z_center[0]), N - np.fabs(z_src-z_center[0])]), axis=0)
        v = A * np.cos(np.pi /4.0 * d/s)
        v[v <= 0] = 0
        h = np.pi/4.0 - d/s
        h[h <= 0] = 0
        h[h > 0] = 1
        return h * v
    else:
        raise "Unimplemented"

@np.vectorize
def template_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-15 *(x - 0.5)))


def template_fitness(I_argmax, sigma, dsigma, size):
    if(len(size) == 1):
        lbound_array = circular_rectified_cosine(np.arange(size[0]), (I_argmax,), 1.0, sigma-dsigma, size)
        ubound_array = template_sigmoid(circular_gaussian(np.arange(size[0]), (I_argmax,), 1.0, sigma+dsigma, size))
        return lbound_array, ubound_array
    elif(len(size) == 2):
        height, width = size
        x = np.arange(width)
        y = np.arange(height)
        xv, yv = np.meshgrid(x, y)
        field_positions = np.zeros((width*height, 2))
        field_positions[:,0] = xv.flatten()
        field_positions[:,1] = yv.flatten()
        lbound_array = circular_rectified_cosine(field_positions, I_argmax, 1.0, sigma-dsigma, size)
        ubound_array = template_sigmoid(circular_gaussian(field_positions, I_argmax, 1.0, sigma+dsigma, size))
        return lbound_array, ubound_array       
    else:
        print("Cannot compute fv fitness in more than 2 dimensions") 
