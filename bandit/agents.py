# -*- coding: utf-8 -*-
import numpy as np
from agents_core import Agent
from utils import my_random_choice
import time

class Bandit_Agent(Agent):
    """ Abstract Agent for a Bandit problem. """
    def __init__(self, n_arms):
        super(Bandit_Agent, self).__init__(None) # no env_shapes
        self.n_arms = n_arms

    def reset(self):
        """ Makes the agent good as new, forgetting previous experiences. """
        raise NotImplementedError("Call to Abstract method in Bandit_Agent")

    def learn(self, a, r):
        """ Overwriting the learn method to restrict it to bandit problems.
        This means there is no state, just action-reward. """
        raise NotImplementedError("Call to Abstract method in Bandit_Agent")

    def act(self):
        """ Overwriting the act method to restrict it to bandit problems
        This means there is no state, just an action. """
        raise NotImplementedError("Call to Abstract method in Bandit_Agent")


class Q_Agent(Bandit_Agent):
    """ Abstract Class that covers the concept of approximating Q values
    through a vector of estimates.
    The default behavior regarding the learning of such estimates is to
    perform a Sample Average. Doesn't define the acting procedure. """

    def __init__(self, n_arms):
        super(Q_Agent, self).__init__(n_arms)
        self.reset()

    def reset(self):
        self._init_Qs() # will be overwritten by optimistic
        self._init_stepsize() # will be overwritten by nonstatic methods

    def _init_Qs(self):
        """ Assigns a first value to the Qs, in the form of a numpy array. """
        self.Qs = np.zeros(self.n_arms) # Q estimates

    def _init_stepsize(self):
        """ Selects the stepsize initialization.
        If changed, _update_stepsize should be changed too. """
        self.Ns = np.zeros(self.n_arms) # number of calls to this action
        self.stepsize = 0

    def _update_stepsize(self, action):
        """ Updates the stepsize given the action. (for static. pass otherwise)
        Default behavior here is 1/N_a """
        self.Ns[action] += 1
        self.stepsize = 1/self.Ns[action]

    def learn(self, a, r):
        """ Sample averaging """
        self._update_stepsize(a)
        self.Qs[a] += self.stepsize*(r - self.Qs[a])
#        if np.random.rand() < 0.001: print("Q values: {}".format(self.Qs))

class EpsGreedy(Q_Agent):
    """ Simple Epsilon Greedy policy over the Q estimates (sample average). """

    def __init__(self, n_arms, epsilon):
        self.epsilon = epsilon
        super(EpsGreedy, self).__init__(n_arms) # inits arms
        self.name = "Stat_EpsGreedy"

    def act(self):
        if np.random.rand()<self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.Qs)

    def tell_specs(self):
        return "eps={}".format(self.epsilon)

class NonStat_EpsGreedy(EpsGreedy):
    """ EpsGreedy agent with constant stepsize = alpha,
    to beat non-stationary bandits. """

    def __init__(self, n_arms, alpha, epsilon=0.1):
        self.alpha = alpha
        super(NonStat_EpsGreedy, self).__init__(n_arms, epsilon)
        self.name = "NonStat_EpsGreedy"

    def _init_stepsize(self):
        """ Fixed stepsize of value alpha """
        self.stepsize = self.alpha

    def _update_stepsize(self, action):
        # actually do nothing !
        pass

    def tell_specs(self):
        return "alpha={}, eps={}".format(self.alpha, self.epsilon)


class OptimisticInits(NonStat_EpsGreedy):
    """ Uses high initial values of Q """

    def __init__(self, n_arms, Q0, alpha=0.1, epsilon=0.):
        self.Q0 = Q0
        super(OptimisticInits, self).__init__(n_arms, alpha, epsilon)
        self.name = "Optimistic"

    def _init_Qs(self):
        self.Qs = np.zeros(self.n_arms) + self.Q0 # Q estimates starting high

    def tell_specs(self):
        return "Q0={}, alpha={}, eps={}".format(self.Q0, self.alpha, self.epsilon)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    init = time.time()
    e_x = np.exp(x - np.max(x))
    if np.random.rand() < 0.00001: print("\t\t\tSingle max&exp time: {}".format(time.time() - init))
    out = e_x / e_x.sum()
    return out

class Gradient_Bandit(Bandit_Agent):
    """ Doesn't estimate the Qs but instead keeps preferences H of each action
    Also keeps a baseline. The alpha parameter refers to the stepsize """

    def __init__(self, n_arms, alpha):
        super(Gradient_Bandit, self).__init__(n_arms)
        self.name = "Gradient"
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.Hs = np.zeros(self.n_arms)
        self.baseline = 0
        self.t = 0 # step number

    def act(self):
        """ Samples from the softmax """
        self.policy = softmax(self.Hs) # save it so we don't recompute
        return my_random_choice(self.n_arms, p=self.policy)

    def learn(self, a, r):
        """ Policy gradient update rule """
        indicator_a = np.zeros(self.n_arms)
        indicator_a[a] = 1.
        self.Hs += self.alpha*(r - self.baseline)*(indicator_a - self.policy)
        # TODO: update baseline
        self.t += 1 # step increment
        self.baseline += (r - self.baseline)/self.t

    def tell_specs(self):
        return "alpha={}".format(self.alpha)


class UCB(Q_Agent):
    """ Keeps both Qs and Us, reward estimates and uncertainty estimates.
    Non stationary algorithm. """

    def __init__(self, n_arms, c, alpha=0.1):
        self.c = c
        self.alpha = alpha
        super(UCB, self).__init__(n_arms)
        self.name = "UCB"

    def reset(self):
        super(UCB, self).reset()
        self.Ns = np.zeros(self.n_arms)
        self.t = 1

    def _init_stepsize(self):
        """ Fixed stepsize of value alpha """
        self.stepsize = self.alpha

    def _update_stepsize(self, action):
        # actually do nothing !
        pass

    def act(self):
        """ UCB selects argmax_a Q_t(a) + c*sqrt(ln(t)/N_a) """
        zeros = np.where(self.Ns == 0)[0]
        if len(zeros) > 0:
            return zeros[0] # arbitrarily the first one we never tried
        else:
            return np.argmax(self.Qs + self.c*np.sqrt(np.log(self.t)/self.Ns))

    def learn(self, a, r):
        """ Sample averaging Qs and incrementing Ns. """
        self.Ns[a] += 1
        self.t += 1
        super(UCB, self).learn(a, r) # Q agent sample averaging (non stat here)

    def tell_specs(self):
        return "c={}".format(self.c)
