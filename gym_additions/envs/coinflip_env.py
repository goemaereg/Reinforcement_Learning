import gym
from gym import spaces
import numpy as np
import random

class CoinFlipEnv(gym.Env):
    """ The player has money {1,..,99} and his goal is to reach 100.
    Each step is a double-or-nothing on the amount of money the player stakes
    as an action.
    The coin has probability p_h to let him win
    States: amount of money {1..99} + terminal states 0 and 100
    Actions: stake {1..min(s,100-s)}
    Dynamics: just money addition of substraction
    Reward: Goal reached (100 money) is +1
    """
    def __init__(self):
        self.max_money = 100
        self.action_space = spaces.Discrete(self.max_money-1) # actually variable
        self.observation_space = spaces.Discrete(self.max_money+1)
        self.reward_space = (0, +1)
        self.p_h = 0.4
        # begin in start state
        self.reset()

    def reset(self):
        self.s = random.choice(self.non_terminal_states()) # initial state is random
        return self.s

    def non_terminal_states(self):
        return list(range(1, self.max_money))

    def terminal_states(self):
        return [0, self.max_money]

    def set_state(self, state):
        """ Sets the balance to a given state. """
        assert state in self.non_terminal_states(), "Invalid state: {}".format(state)
        self.s = state

    def legal_actions(self, s=None):
        if s is None:
            s = self.s

        return list(range(1, min(s, self.max_money-s)+1))


    def p(self, s, a):
        """ Dynamics function p(s',r | s,a).
        Returns a dictionary where d[s', reward]: probability
        Where the sum over the dictionary is 1. """
        assert s in self.non_terminal_states(), "Asking p for invalid, terminal state: {}".format(s)
        assert a in self.legal_actions(s), "Invalid action: {} for state {}".format(a,s)
        r = int(s+a >= self.max_money)
        d = {}
        d[s+a, r] = self.p_h
        d[s-a, 0] = 1-self.p_h
        return d

    def step(self, action):
        """ Substracts or adds money based on the coin flip """
        # Just flip it, just flip it
        assert action in self.legal_actions(), "Invalid action {} in state {}".format(action,self.s)
        gain = 2*int(np.random.rand()<self.p_h) - 1 # -1 or +1
        self.s += gain*action # double or nothing stake
        if self.s >= self.max_money:
            return self.max_money, 1, 1, {}
        elif self.s <= 0:
            return 0, 0, 1, {}
        else:
            return self.s, 0, 0, {}

    def render(self):
        print("Balance: {}".format(self.s))
