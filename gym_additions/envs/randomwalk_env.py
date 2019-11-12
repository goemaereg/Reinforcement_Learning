import gym
from gym import spaces
import numpy as np

class RandomWalkEnv(gym.Env):
    """ 5 states in a line as presented in RL book.
    This is a MRP, there are no actions, the goal is prediction
    aka evaluation of the state values.
    States: 1 to 5 (A to E) + terminal states left and right (0 and 6)
    Dynamics: randomly move right or left p=1/2.
    Reward: reaching state 5 (rightmost)
    The values can be computed exactly and are (1/6 2/6 3/6 4/6 5/6). """

    def __init__(self):
        self.action_space = None
        self.n_s = 5
        self.observation_space = spaces.Discrete(self.n_s+2) # 2 terminals
        # begin in start state

    def terminal_states(self):
        return [0, self.n_s +1]

    def non_terminal_states(self):
        return range(1, self.n_s+1)

    def p(self, s):
        """ Dynamics function that outputs the dictionary p(s',r|s a=None)"""
        assert s in self.non_terminal_states(), "Dynamics: non terminal states only"
        d = {}
        r = int(s==self.n_s) # we are on rightmost
        d[s+1, r] = 1/2
        d[s-1, 0] = 1/2
        return d
