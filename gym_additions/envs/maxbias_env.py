import gym
from gym import spaces
import numpy as np

class MaxBiasEnv(gym.Env):
    """ Environment designed to show the maximization bias all the Sarsa methods
    suffer from (Sarsa QLearning ExpectedSarsa)
    States: A, B, terminal
    Dynamics & Actions: Starting from A, action right (1) finishes the episode
        with reward 0. Left (0) leads to B with reward 0, from where any action
        leads to the terminal state with reward ~N(-0.1, 1)
    Hence the optimal behavior is always to go right from A.
    """

    def __init__(self):
        # We choose arbitrarily the number of actions from B out
        self.n_actions = 6 # might be too low to observe the effect
        self.action_space = spaces.Discrete(self.n_actions)
        self.n_s = 2
        self.observation_space = spaces.Discrete(self.n_s+1) # terminal: 2
        # begin in start state
        self.reset()

    def reset(self):
        self.s = 1
        return self.s

    def step(self, action):
        """ Moves the agent in the action direction."""
        r = 0
        if self.s == 1: # State A
            self.s += 2*(action%2) - 1 # {left, right}={0,1} -> {-1, 1}
        elif self.s == 0: # State B
            self.s = 2  # any action is terminal
            r = np.random.randn() - 0.1
        # Returns
        done = (self.s==2)
        return self.s, r, done, {}

    def render(self):
        if self.s == 1:
            print("State (A)")
        elif self.s == 0:
            print("State (B)")
        else:
            print("State [T]")
