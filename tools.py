from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import random
import matplotlib.pyplot as plt
import itertools

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

def cconv_step(fu, w_params):
    # Reminder : the Step function is here defined as :
    #a stepwise : w(x) = Ae 1_(|x| < ke si) - ki Ae 1_(|x| < si)
    # 1 params : Ae >= 0 ; ke in [0, 1], ki in [0,1] ,  si > 0
    
    Ae, ke, ki, si = w_params
    se = ke * si
    Ai = ki * Ae

    N = fu.shape[0]

    lat = np.zeros((N,))
    # We need to compute the lateral contribution at the first position 
    # with an order of N sums and 2 products            
    # We identify the corner points of the steps
    # which are the ones for which the lateral contribution will change as we move  
    # along a sliding window
    #          -------
    #          |     |
    # ---------=     =---------
    # On the above step function, the corner points are denoted by "="
    # In corners[0] is the leftmost, corners[1] is the rightmost
    # Mind that we use a toric topology, so that for the first unit 
    # the leftmost corner is actually on the right because of the wrap around

    # we must handle the specific cases where the excitatory or inhibitory
    # radii cover the whole field
    
    all_excitatories = se >= N-1
    if not all_excitatories:
        corners_exc = [int(np.ceil(N-se)), int(np.floor(se))]
        lat_e = Ae * (sum(itertools.islice(fu, 0, corners_exc[1]+1)) + sum(itertools.islice(fu,corners_exc[0],None)))
    else:
        lat_e = Ae * sum(fu)

    all_inhibitories = si >= N-1
    if not all_inhibitories:
        corners_inh = [int(np.ceil(N-si)), int(np.floor(si))]
        lat_i = Ai * (sum(itertools.islice(fu,0, corners_inh[1]+1)) + sum(itertools.islice(fu,corners_inh[0], None)))
    else:
        lat_i = Ai * sum(fu)

    lat[0] = lat_e - lat_i

    for i in xrange(1, N):
        # We can then compute the lateral contributions of the other locations
        # with a sliding window by just updating the contributions
        # at the "corners" of the weight function
        if not all_excitatories:
            lat_e -= Ae * fu[corners_exc[0]]
            corners_exc[0] += 1
            if(corners_exc[0] >= N):
                corners_exc[0] = 0
        
            corners_exc[1] += 1
            if(corners_exc[1] >= N):
                corners_exc[1] = 0
            lat_e += Ae * fu[corners_exc[1]]

        if not all_inhibitories:
            lat_i -= Ai * fu[corners_inh[0]]
            corners_inh[0] += 1
            if(corners_inh[0] >= N):
                corners_inh[0] = 0
        
            corners_inh[1] += 1
            if(corners_inh[1] >= N):
                corners_inh[1] = 0
            lat_i += Ai * fu[corners_inh[1]]

        lat[i] = lat_e - lat_i
    return lat


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
