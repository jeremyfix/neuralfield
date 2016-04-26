#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import sys
import os
import time
import libPyPopot

import dnf
import scenario
import tools

if(len(sys.argv) != 4):
    print("Usage %s weight trial_nb scenario" % sys.argv[0])
    print("weight in [dog, doe, dol, step]")
    print("scenario in [selection, wm]")
    sys.exit(-1)


weight = sys.argv[1]
if(not weight in ['dog', 'doe', 'dol','step']):
    raise "Unrecognized weight function '%s'" % weight

trial_nb = int(sys.argv[2])
N = 100 # The number of positions
size = (N, )

dnf_builder = lambda:dnf.DNF(size, weight, dnf.heaviside_tf)

scenario_name = sys.argv[3]
if(not scenario_name in ['selection', 'wm']):
    raise "Unrecognized scenario %s" % scenario_name


if(scenario_name == 'selection'):

    scenario.CompetitionScenario.def_scenario(0.4, size)
    scenario.CompetitionScenario.def_scenario(0.6, size)
    scenario.CompetitionScenario.def_scenario(0.8, size)

    pso_max_epoch = 100
    swarm_size = 25

    # size is supposed to be defined globally
    def evaluate(params):
        fit = 0.0
        field = dnf_builder()
        field.set_params(params)
        for i in range(len(scenario.CompetitionScenario.inputs)):
            scenar = scenario.CompetitionScenario(i)
            field.reset()
            while(not scenar.is_finished()):
                field.step(scenar.get_input())
                fit += scenar.get_fitness(field.get_output())
                scenar.step()
        return fit

    def plot_best(test_params, filename=None):
        field = dnf_builder()
        field.set_params(test_params)
        fig = plt.figure()

        gs = gridspec.GridSpec(len(scenario.CompetitionScenario.inputs), 2)
        for i in range(len(scenario.CompetitionScenario.inputs)):
            scenar = scenario.CompetitionScenario(i)   
            fuhistories = []
            field.reset()
            while(not scenar.is_finished()):
                field.step(scenar.get_input())
                scenar.step()
                fuhistories.append(field.get_output())
            fuhistories = np.array(fuhistories)
         
            ax = fig.add_subplot(gs[2*i])
            ax.imshow(scenar.get_full_input(), cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
            ax.set_aspect('auto')

            ax = fig.add_subplot(gs[2*i + 1])
            ax.imshow(fuhistories, cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
            ax.set_aspect('auto')


        if(filename):
            plt.savefig(filename)
        else:
            plt.show()
        
else:
    scenario.WorkingMemoryScenario.init(size)

    pso_max_epoch = 1000  
    swarm_size = 200

    # size is supposed to be defined globally
    def evaluate(params):
        fit = 0.0
        field = dnf_builder()
        field.set_params(params)
        
        scenar = scenario.WorkingMemoryScenario()
        while(not scenar.is_finished()):
            field.step(scenar.get_input())
            fit += scenar.get_fitness(field.get_output())
            scenar.step()
        return fit

    def plot_best(test_params, filename=None):
        field = dnf_builder()
        field.set_params(test_params)
        scenar = scenario.WorkingMemoryScenario()   
        fuhistories = []
        field.reset()
        while(not scenar.is_finished()):
            field.step(scenar.get_input())
            scenar.step()
            fuhistories.append(field.get_output())
        fuhistories = np.array(fuhistories)
         
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(scenar.get_full_input(), cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
        ax.set_aspect('auto')

        ax = fig.add_subplot(122)
        ax.imshow(fuhistories, cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
        ax.set_aspect('auto')


        if(filename):
            plt.savefig(filename)
        else:
            plt.show()


#Seed initialization
libPyPopot.seed((int)((time.time()%100)*10000))

########################### Lower and upper bounds for the parameters
# dt_tau ; h
lbounds = [0,  -1]
ubounds = [0.3, 1]

#            Ae     ks       ka     si
lbounds += [ 0  ,   0.001,     0. ,  1.0]
ubounds += [ 5  ,   1. ,     1.0 ,  N]

#if weight == 'step':
#    #            k
#    lbounds += [ 0 ]
#    ubounds += [ 1 ]


#Set the dimension of our problem
dimension = len(lbounds)

def lbound(index):
    return lbounds[index]

def ubound(index):
    return ubounds[index]

def stop(fitness, epoch):
    return (epoch >= pso_max_epoch) #or (fitness <= 1e-4) 

########## Define some directory for storing the results
if not os.path.isdir('./files/%s-%s/' % (scenario_name, weight)):
	os.makedirs('./files/%s-%s/' % (scenario_name, weight))
if not os.path.isdir('./images/%s-%s/' % (scenario_name, weight)):
	os.makedirs('./images/%s-%s/' % (scenario_name, weight))


########## For optimization
print("Running the optimization in 1D")
algo = libPyPopot.SPSO_2006(swarm_size, dimension, lbound, ubound, stop, evaluate)
file_fitness = open('files/%s-%s/fitness_%05d.data' % (scenario_name, weight,trial_nb),'w')
file_fitness.close()
while(not(stop(algo.bestFitness(), algo.getEpoch()))):
    algo.step()
    file_fitness = open('files/%s-%s/fitness_%05d.data' % (scenario_name, weight,trial_nb),'a')
    file_fitness.write("%i %f %s\n" % (algo.getEpoch(),algo.bestFitness(), " ".join(map(str,algo.bestParticle()))))
    file_fitness.close()

    sys.stdout.write("\r " + ("%i %f" % (algo.getEpoch(),algo.bestFitness())).ljust(70))
    sys.stdout.flush()
    #algo.run(1)
sys.stdout.write('\n')

test_params = algo.bestParticle()
print("Best solution [fitness=%f]: " % evaluate(test_params), test_params)

# Plotting the best
plot_best(test_params, 'images/%s-%s/%05i.png'%(scenario_name, weight,trial_nb))
#plot_best(test_params)


