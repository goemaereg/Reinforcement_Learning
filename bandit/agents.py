import numpy as np
from agents_core import Agent

class Bandit_Agent(object):
    """ Abstract Agent for a Bandit problem which keeps track of estimates for Q.
        """
    def __init__(self, n_arms):
        super(Bandit_Agent, self).__init__()
        self.n_arms = n_arms
        self.estimates = self.initialize_estimates() # numpy array

    def initialize_estimates(self):
        """ Assigns a first value to the Qs, in the form of a numpy array.
            """
        raise NotImplementedError("Call to Abstract method in Bandit_Agent")


    def epsilon_greedy(epsilon, Q):
        """ Simple epsilon-greedy strategy given action-value estimator Q
            """
        return np.random.randint(len(Q)) if np.random.rand() < epsilon \
                                         else np.argmax(Q)
