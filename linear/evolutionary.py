""" Evolutionary agents (Genetic algos, Evolution strategies...) expressed
as learning agents similarly to the reinforcement learning paradigm.
Each individual is evaluated one by one and the generation is updated when the
whole population was evaluated.
"""
import numpy as np
from utils import softmax, assert_not_abstract, my_random_choice, my_argmax, prod
from agents_core import Agent
import random
from collections import deque
import warnings
#warnings.filterwarnings("error")



class GeneticAlgorithm(Agent):
    """ RL adaptation of a simple Genetic Algorithm with no crossover. """
    def __init__(self, env_shapes, N, std, mu, **kwargs):
        super(GeneticAlgorithm, self).__init__(env_shapes) # inits shapes
        self.name = 'GeneticAlgorithm'
        self.input_dim = prod(self.input_shape) # flattens the shape
        print((self.input_dim, self.n_actions))
        self.lmbda = N  # number of individuals (in the litterature)
        self.std = std  # standard deviation of mutations
        self.mu = mu      # k best individuals are selected
        self.reset()

    def reset(self):
        self.population = [ # set of individuals i.e. (small) weight vectors
            np.random.randn(self.input_dim, self.n_actions)*0.01
            for i in range(self.lmbda)
        ]
        self.fitnesses = np.empty(self.lmbda)
        self.current_fitness = 0
        self.current_indiv = 0 # currently evaluated individual

    def policy(self, w, x, a=None):
        """ Q function approximator - in this case, linear -
        for a weight vector w. Applies a softmax to the results"""
        policy_x = np.dot(w.T,x)
        return softmax(policy_x)

    def act(self, x):
        """ The current individual acts. """
        current_w = self.population[self.current_indiv]
        return my_argmax(self.policy(current_w, x))

    def learn(self, x, a, r, x_, d):
        """ Updates the fitness; if the episode is over, switches to next.
        If the whole generation is done, compute next weights (individuals).
        Selection is uniform over best mu"""
        self.current_fitness += r
        if d:
            self.fitnesses[self.current_indiv] = self.current_fitness
            self.current_indiv += 1
            self.current_fitness = 0
            if self.current_indiv == self.lmbda: # pop maxed
                best_indivs = np.argpartition(self.fitnesses, -self.mu)[-self.mu:]
                if np.random.randn()<0.01: print("\tNew gen, fits={}".format(self.fitnesses))
                new_population = []
                for i in range(self.lmbda):
                    parent   = best_indivs[np.random.randint(self.mu)]
                    parent_w = self.population[parent]
                    mutation = np.random.standard_normal(parent_w.shape)*self.std
                    new_population.append(parent_w + mutation)
                self.population = new_population
                self.current_indiv = 0

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "popsize={}, std={}, mu={}"\
               .format(self.lmbda, self.std, self.mu)

class EvolutionStrategies(Agent):
    """ Simple (mu, lambda)-ES, coded from an RL's mindset """
    def __init__(self, env_shapes, N, std, mu, **kwargs):
        super(EvolutionStrategies, self).__init__(env_shapes) # inits shapes
        self.name = 'EvolutionStrategies'
        self.input_dim = prod(self.input_shape) # flattens the shape
        print((self.input_dim, self.n_actions))
        self.lmbda = N  # number of individuals (in the litterature)
        self.mu  = mu   #
        self.std = std  # standard deviation of distribution
        self.reset()

    def reset(self):
        self.mean = np.random.randn(self.input_dim, self.n_actions)*self.std
        self.population = np.array([ # set of individuals i.e. (small) weight vectors
            self.mean + np.random.randn(self.input_dim, self.n_actions)*self.std
            for i in range(self.lmbda)
        ])
        self.fitnesses = np.empty(self.lmbda)
        self.current_fitness = 0
        self.current_indiv = 0 # currently evaluated individual

    def policy(self, w, x, a=None):
        """ Q function approximator - in this case, linear -
        for a weight vector w. Applies a softmax to the results"""
        policy_x = np.dot(w.T,x)
        return softmax(policy_x)

    def act(self, x):
        """ The current individual acts. """
        current_w = self.population[self.current_indiv]
        return my_random_choice(range(self.n_actions), self.policy(current_w, x))

    def learn(self, x, a, r, x_, d):
        """ Updates the fitness; if the episode is over, switches to next.
        If the whole generation is done, compute next weights (individuals).
        Selection is uniform over best k"""
        self.current_fitness += r
        if d:
            self.fitnesses[self.current_indiv] = self.current_fitness
            self.current_indiv += 1
            self.current_fitness = 0
            if self.current_indiv == self.lmbda: # pop maxed
                best_indivs = np.argpartition(self.fitnesses, -self.mu)[-self.mu:]
                if np.random.randn()<0.01: print("\tNew gen, fits={}".format(self.fitnesses))
                self.mean = self.population[best_indivs].mean(axis=0)
                self.population = np.array([
                    self.mean + np.random.randn(self.input_dim, self.n_actions)*self.std
                    for i in range(self.lmbda)
                ])
                self.current_indiv = 0

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "popsize={}, std={}, mu={}"\
               .format(self.lmbda, self.std, self.mu)
