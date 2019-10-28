import gym
from gym import spaces
import numpy as np

class Cross6Env(gym.Env):
    """ 6 states in a line as presented in Hierarchical Deep Reinforcement
    Learning: Integrating Temporal Abstraction and Intrinsic Motivation.
    Initial state: tarting at state 2,
    States: 1 to 6, but states are duplicated to carry reached-6 info
    Actions: left&right
    Dynamics: going left is deterministic, going right is 1/2 success 1/2left
    Reward: Going to s1 gives a reward of 1 if s6 was reached 0.01 otherwise
        """
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(6), # stqtes in a line
                spaces.Discrete(2)  # boolean reachedS6
                ))
        self.moves = {
                0: -1,  # left
                1: 1,   # right
                }
        self.rewards = {
                0: 0,    # non terminal state
                1: 0.01, # didn't reach S6
                2: 1     # reached S6
        }

        # begin in start state
        self.reset()

    def step(self, action):
        """ Moves the agent in the action direction.
            """
        # Next, moving according to action
        move = self.moves[action]
        if (move==1) and (np.random.rand()<0.5):
            move = -1
        self.S += move
        # handling the reward condition of reaching S6
        if self.S >= 6:
            self.reachedS6 = 1

        # Finally, setting the agent back into the grid if fallen out
        self.S = min(6, self.S)

        # Returns; anything brings a reward of -1
        if self.S <= 1:
            return (self.S-1, self.reachedS6), self.rewards[1+self.reachedS6], 1, {}
        else:
            return (self.S-1, self.reachedS6), 0, 0, {}

    def reset(self):
        self.S = 2 # initial at S2
        self.reachedS6 = 0 # 1 as soon as we reach S6
        return (self.S-1, self.reachedS6)

    def render(self):
        s = np.zeros(6, dtype=int).astype(str)
        s[0] = 'T'
        s[5] = 'R' # reward
        s[self.S-1] = '.'
        print(str(s) + " {}".format((self.S-1, self.reachedS6)))
