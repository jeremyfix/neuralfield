import numpy as np
import tools

''' Script for defining a 1D neural field 
    The equation reads:

    U(x, t+1) = U(x,t) + dt_tau * (-U(x,t) + [w o f(V)](x,t) + I(x,t) + h)

    where [f o g](x) denotes the circular convolution of f and g taken in x

    f is a heaviside : f(x) = 1_[x>=0]
    There are 2+nw free parameters : [dt_tau, h]
    with nw parameters for the weights; these weights being:

      * a DOG : w(x) = Ae exp(-x**2 / (2 * (ks*si)**2)) - ka Ae exp(-x**2 / (2 si**2))
         4 params : Ae >= 0 ; ks in [0, 1], ka in [0, 1], si > 0

      * a DOE : w(x) = Ae exp(-4 |x| / (ks*si)**2) - ka Ae exp(-4 |x| / si**2)
         4 params : Ae >= 0 ; ks in [0, 1], ka in [0, 1], si > 0

      * a DOL : w(x) = Ae [1 - |x|/(2 ks si)]^+ - ka Ae [1 - |x| / (2 si)]^+
         4 params : Ae >= 0 ; ks in [0, 1] ; ka in [0, 1] ; si > 0

      * a stepwise : w(x) = Ae 1_(|x| < ks si) - ka Ae 1_(|x| < si) - k ka Ae 1_(|x| >= si)
         5 params : Ae >= 0 ; ks in [0, 1], ka in [0,1] , k in [0, 1], si > 0

    The equation is simulated synchronously, with Euler and a time step dt
'''

def heaviside_tf(x):
    return 0.5 * (np.sign(x) + 1)

def sigmoid_tf(x,a=1,b=-1,x0=0):
    return a / (1.0 + np.exp(b*(x-x0)))

def rectified_tf(x):
    y = x.copy()
    y[y < 0] = 0.0
    return y

def dog(dx, params):
    Ae, ke, ki, si = params
    return Ae * np.exp(-dx**2/(2.0 * (ke*si)**2)) - ki * Ae * np.exp(-dx**2/(2.0 * si**2))

def doe(dx, params): 
    Ae, ke, ki, si = params
    return Ae * np.exp(-4.0 * np.fabs(dx)/((ke*si)**2)) - ki * Ae * np.exp(-4.0 * np.fabs(dx)/(si**2))

def pos_part_s(x):
    if(x >= 0):
        return x
    else:
        return 0
pos_part = np.vectorize(pos_part_s, otypes=[np.float])

def dol(dx, params):
    Ae, ke, ki, si = params
    return Ae * pos_part(1.0 - np.fabs(dx)/(2.0 * ke*si)) - ki * Ae * pos_part(1.0 - np.fabs(dx)/(2.0 * si))

def step(dx, params):
    Ae, ke, ki, si = params
    return Ae * np.piecewise(dx, [np.fabs(dx) < ke*si], [1]) - ki * Ae * np.piecewise(dx, [np.fabs(dx) < si], [1]) #- k *ki * Ae * np.piecewise(dx, [np.fabs(dx) >= si], [1])

class DNF:
    
    def __init__(self, size, weights_name, tf):
        if(not weights_name in ["dog", "doe", "dol", "step"]):
            raise '''Unrecognized weight kernel, options are
            - dog : difference of gaussians
            - doe : difference of exponentiels
            - dol : difference of linear functions
            - step : step wise function
            '''
        self.weights_name = weights_name
        self.dict_weights = {'dog' : dog, 'dol': dol, 'doe': doe, 'step': step}
        if(not self.weights_name in self.dict_weights.keys()):
            raise "Unknown weight function '%s'" % self.weights_name

        self.transfer_function = tf

        self.size = size
        self.u = np.zeros(size)
        self.fu = self.transfer_function(self.u)

        self.dt_tau = 0
        self.h = 0
        self.w = np.zeros(self.size)

    def __build_weights(self, wparams):
        self.w = np.arange(self.size[0])
        for i in range(self.size[0]):
            self.w[i] = np.min([self.w[i], self.size[0] - np.fabs(self.w[i])])
        self.w = self.dict_weights[self.weights_name](self.w, wparams)

    def reset(self):
        self.u = np.zeros(self.size)
        self.fu = self.transfer_function(self.u)        

    def set_params(self, params):
        self.dt_tau = params[0]
        self.h = params[1]
        self.__build_weights(params[2:])

    def step(self, I):
        self.u = self.u + self.dt_tau * (-self.u + tools.cconv(self.fu, self.w)  + I + self.h)
        self.fu = self.transfer_function(self.u)

    def get_output(self):
        return self.fu.copy()

if(__name__ == '__main__'):
    print("Testing")
