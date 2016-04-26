
import numpy as np
import tools
import matplotlib.pyplot as plt

class Scenario:

    def __init__(self, size):
        self.t = 0
        self.size = size
        self.I = np.zeros(size)

    def is_finished(self):
        return True

    def get_input(self):
        return self.I

    def step(self):
        self.t += 1

    def get_fitness(self, fu):
        return 0.0


class CompetitionScenario:

    '''The static components of the scenario'''
    size = 0
    inputs = []
    x_weaks = []
    x_strong = 0
    amplitudes = []
    sigma = 4.0
    sigmat = 5.0
    Tmax = 40

    @classmethod
    def def_scenario(cls, A, s):
        cls.size = s
        cls.width = cls.size[0]
        cls.x_weaks = [cls.width/10.0, 5.0 * cls.width/10.0, 7.0 * cls.width/10.0, 9.0 * cls.width/10.0]
        cls.x_strong = 3.0 * cls.width/10.0
        cls.amplitudes.append({'weak': A, 'strong':A+0.2})
        cls.generate_inputs(A, A+0.2)

    @classmethod
    def generate_inputs(cls, A_weak, A_strong):
        width = cls.size[0]
        I = np.zeros((cls.Tmax, width))
        for t in range(cls.Tmax):
            if(t <= 0.5 * cls.Tmax):
                width = cls.size[0]
                for x in cls.x_weaks:
                    I[t,:] += tools.gaussian(x, cls.sigma, A_weak, cls.size) 
                I[t,:] += tools.gaussian(cls.x_strong, cls.sigma, A_strong, cls.size) 
            else:
                width = cls.size[0]
                scale = np.exp(-(t-0.5 * cls.Tmax)/cls.sigmat)
                for x in cls.x_weaks:
                    I[t,:] += tools.gaussian(x, cls.sigma, scale * A_weak, cls.size) 
                I[t,:] += tools.gaussian(cls.x_strong, cls.sigma, scale * A_strong, cls.size)       
        cls.inputs.append(I)


    def __init__(self, scenario_nb):
        self.t = 0
        self.scenario_nb = scenario_nb
        if(self.scenario_nb < 0 or self.scenario_nb >= len(self.__class__.inputs)):
            raise "You should peak a scenario nb in [0, %i] and not %i " % (len(self.__class__.inputs)-1, scenario_nb)

    def is_finished(self):
        return self.t >= self.__class__.Tmax

    def get_input(self):
        return self.__class__.inputs[self.scenario_nb][self.t,:]

    def get_full_input(self):
        return self.__class__.inputs[self.scenario_nb].copy()

    def step(self):
        self.t += 1

    def get_fitness(self, fu):
        return self.get_fitness_heaviside(fu)

    '''!!!!! I suppose the activities to be in [0, 1] !!!!'''
    def get_fitness_heaviside(self, fu):
        f = 0.0
        if(int(self.__class__.Tmax/2) - 5 <= self.t <= int(self.__class__.Tmax/2)):
            for i in range(self.__class__.size[0]):
                if(abs(i - self.__class__.x_strong) <= self.__class__.sigma):
                    f += 1 - fu[i]
                else:
                    f += fu[i]
        elif(self.t >= self.__class__.Tmax-1):
            f += fu.sum()
        return f
          
    def get_fitness_template(self, fu):
        f = 0.0
        if(self.t == int(self.__class__.Tmax/2)):
            lb, ub = tools.template_fitness(self.__class__.x_strong, sigma, 0.0, self.__class__.size)
            values = np.zeros(self.__class__.size)
            where_larger_than_ub = (fu > ub)
            where_smaller_than_lb = (fu < lb)
            values[where_larger_than_ub] =  ((ub - fu)[where_larger_than_ub]).flatten()
            values[where_smaller_than_lb] =  (fu - lb)[where_smaller_than_lb]        
            f = np.sum(values.flatten()**2)
        elif(self.t == self.__class__.Tmax-1):
            f = np.sum(fu**2)
        return f

    def plot_templates(self, fu):
        lb, ub = tools.template_fitness(x_strong, sigma, 0.0, size)
        plt.figure()
        plt.plot(lb, 'k--')
        plt.plot(ub, 'k--')
        #plt.plot(fu)
        plt.show()


class WorkingMemoryScenario:

    '''The static components of the scenario'''
    Tselected0 = 30
    Tselected1 = 50
    Tselection = (Tselected1 - Tselected0)/2

    Tweak = 20
    Tmove = 80
    Tfinal = 40
    Tmax = Tselected1 + Tselection/2 + Tweak + Tmove + Tfinal

    A_weak = 0.3
    A_strong = 1.0
    sigma = 4.0    
    
    @classmethod
    def init(cls, s):
        cls.size = s
        cls.width = cls.size[0]
        cls.x0 = cls.width/5.
        cls.x1 = 3*cls.width/5.
        cls.dx = cls.width/5.
        cls.generate_inputs()
        print("The working scenario will last %i steps" % cls.Tmax)
    
    @classmethod
    def generate_inputs(cls):
        cls.input = np.zeros((cls.Tmax, cls.width))
        for t in range(cls.Tmax):
            if(t <= cls.Tselected0 - cls.Tselection/2):
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, cls.A_weak, cls.size) + \
                                  tools.gaussian(cls.x1, cls.sigma, cls.A_weak, cls.size)
            elif(t <= cls.Tselected0):
                ti = cls.Tselected0 - cls.Tselection/2
                tf = cls.Tselected0 
                A = (cls.A_strong - cls.A_weak) / (tf - ti) * (t - ti) + cls.A_weak
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, A, cls.size) + \
                                  tools.gaussian(cls.x1, cls.sigma, cls.A_weak, cls.size)
            elif(t <= cls.Tselected0 + cls.Tselection/2):
                ti = cls.Tselected0
                tf = cls.Tselected0 + cls.Tselection/2
                A = (cls.A_weak - cls.A_strong) / (tf - ti) * (t - ti) + cls.A_strong
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, A, cls.size) + \
                                  tools.gaussian(cls.x1, cls.sigma, cls.A_weak, cls.size)
            elif(t <= cls.Tselected1 - cls.Tselection/2):
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, cls.A_weak, cls.size) + \
                                  tools.gaussian(cls.x1, cls.sigma, cls.A_weak, cls.size)
            elif(t <= cls.Tselected1):
                ti = cls.Tselected1 - cls.Tselection/2
                tf = cls.Tselected1 
                A = (cls.A_strong - cls.A_weak) / (tf - ti) * (t - ti) + cls.A_weak
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, cls.A_weak, cls.size) + \
                                  tools.gaussian(cls.x1, cls.sigma, A, cls.size)
            elif(t <= cls.Tselected1 + cls.Tselection/2):
                ti = cls.Tselected1
                tf = cls.Tselected1 + cls.Tselection/2
                A = (cls.A_weak - cls.A_strong) / (tf - ti) * (t - ti) + cls.A_strong
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, cls.A_weak, cls.size) + \
                                  tools.gaussian(cls.x1, cls.sigma, A, cls.size)
            elif(t <= cls.Tselected1 + cls.Tselection/2 + cls.Tweak):
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, cls.A_weak, cls.size) + \
                                  tools.gaussian(cls.x1, cls.sigma, cls.A_weak, cls.size)
            elif(t <= cls.Tselected1 + cls.Tselection/2 + cls.Tweak + cls.Tmove):
                ti = cls.Tselected1 + cls.Tselection/2 + cls.Tweak
                tf = cls.Tselected1 + cls.Tselection/2 + cls.Tweak + cls.Tmove
                x1_t = cls.dx/(tf - ti) * (t - ti) + cls.x1 
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, cls.A_weak, cls.size) + \
                                  tools.gaussian(x1_t, cls.sigma, cls.A_weak, cls.size)
            else:
                # Remove the bumps
                ti = cls.Tselected1 + cls.Tselection/2 + cls.Tweak + cls.Tmove
                scale = np.exp(-(t - ti)/10.0)
                cls.input[t,:] += tools.gaussian(cls.x0, cls.sigma, scale *cls.A_weak, cls.size) + \
                                  tools.gaussian(cls.x1 + cls.dx, cls.sigma, scale *cls.A_weak, cls.size)

    
    def __init__(self):
        self.t = 0

    def is_finished(self):
        return self.t >= self.__class__.Tmax

    def get_input(self):
        return self.__class__.input[self.t,:]

    def get_full_input(self):
        return self.__class__.input.copy()

    def step(self):
        self.t += 1

    def get_fitness(self, fu):
        return self.get_fitness_heaviside(fu)

    '''!!!!! I suppose the activities to be in [0, 1] !!!!'''
    def get_fitness_heaviside(self, fu):
        f = 0.0
        cls = self.__class__
        if(self.t == cls.Tselected0 - cls.Tselection/2):
            # No bumps
            f = np.sum(fu)
        elif(self.t == cls.Tselected0 + cls.Tselection/2):
            # A single bump at x0 of width sigma
            for i in range(cls.size[0]):
                if(abs(i - cls.x0) <= cls.sigma):
                    f += 1 - fu[i]
                else:
                    f += fu[i]
        elif(self.t == cls.Tselected1 + cls.Tselection/2):
            # Two bumps, one at x0, the other at x1
            for i in range(cls.size[0]):
                if(abs(i - cls.x0) <= cls.sigma or abs(i - cls.x1) <= cls.sigma):
                    f += 1 - fu[i]
                else:
                    f += fu[i]
        elif(self.t == cls.Tselected1 + cls.Tselection/2 + cls.Tweak):
            # Still two bumps at x0 and x1
            for i in range(cls.size[0]):
                if(abs(i - cls.x0) <= cls.sigma or abs(i - cls.x1) <= cls.sigma):
                    f += 1 - fu[i]
                else:
                    f += fu[i]
        elif(self.t == cls.Tselected1 + cls.Tselection/2 + cls.Tweak + cls.Tmove):
            # Two bumps at x0 and x1+dx
            for i in range(cls.size[0]):
                if(abs(i - cls.x0) <= cls.sigma or abs(i - (cls.x1 + cls.dx)) <= cls.sigma):
                    f += 1 - fu[i]
                else:
                    f += fu[i]
        elif(self.t == cls.Tmax-1):
            # No bumps
            f = np.sum(fu)
            
        return f


if(__name__ == '__main__'):

    # Testing and plotting the scenarios

    size = (100,)

    # Competition Scenario
    CompetitionScenario.def_scenario(0.4, size)
    CompetitionScenario.def_scenario(0.6, size)
    CompetitionScenario.def_scenario(0.8, size)

    s0 = CompetitionScenario(0)
    s1 = CompetitionScenario(1)
    s2 = CompetitionScenario(2)

    fig = plt.figure()
    ax = fig.add_subplot(141)
    plt.imshow(s0.get_full_input(), cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
    plt.colorbar()
    ax.set_aspect('auto')

    ax = fig.add_subplot(142)
    plt.imshow(s1.get_full_input(), cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
    plt.colorbar()
    ax.set_aspect('auto')

    ax = fig.add_subplot(143)
    plt.imshow(s2.get_full_input(), cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
    plt.colorbar()
    ax.set_aspect('auto')


    # Working memory
    WorkingMemoryScenario.init(size)
    s0 = WorkingMemoryScenario()

    ax = fig.add_subplot(144)
    plt.imshow(s0.get_full_input(), cmap=plt.cm.gray_r, origin='lower', clim=[0, 1])
    plt.colorbar()
    ax.set_aspect('auto')


    plt.show()
