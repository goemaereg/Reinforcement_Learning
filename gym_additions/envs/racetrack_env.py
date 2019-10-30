import gym
from gym import spaces
import numpy as np
import random


class RaceTrackEnv(gym.Env):
    """ Discrete version of a car race with a simple right turn, as in RLbook.
    The car starts at the Start line, lower corner of the world,
    and runs its way up to the upper right corner of the world.
    A subrectangle of the world (lower right corner) is an obstacle and
    shoots you right back at the starting line
    States: Position as a pair [[0,size]]^2 x velocity as a pair [[0,5]]^2.
    Init: Positions [[0,random]]
    Actions: {-1,0,1}^2 for velocity increment; or 1D equivalent.
    Rewards: -1 per timestep.
        """
    def __init__(self):
        self.size = 9
        self.action_space = spaces.Discrete(9) # see action2move
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.size), # position
                spaces.Discrete(self.size), # position
                spaces.Discrete(5), # velocity
                spaces.Discrete(5)  # velocity
                ))
        # begin in start state (supposedly called in the code)
        self.obstacle = np.array([2*self.size//3, self.size//3])
        print("World size: {}; obstacle: {}".format(self.size, self.obstacle))
        self.reset()

    def _action2move(self, action):
        """ Converts an action [[0,8]] to a move [[-1,1]]^2 """
        row = action//3
        return np.array([row, action - 3*row])-1

    def _move2action(self, move):
        move += 1
        return 3*move[0] + move[1]

    def reset(self):
        self.position = np.array([0, np.random.randint(self.obstacle[1])], dtype=int)
        self.speed = np.zeros(2, dtype=int)
        self.pos_history = [tuple(self.position)] # for rendering
        return (*tuple(self.position), *tuple(self.speed))

    def _hit_obstacle(self, position=None):
        """ Returns whether the car hit borders """
        if position is None:
            position = self.position
        # First, the outer edges
        if np.any(position<0) or np.any(position>=self.size):
            # print("Position {} hit the world border".format(position))
            return True
        # Now the obstacle
        diff_pos = position - self.obstacle
        if (diff_pos[0]<=0) and (diff_pos[1]>=0):
            # print("Position {} hit the obstacle".format(position))
            return True
        # ... else,
        # print("NP with Position {}".format(position))
        return False

    def step(self, action):
        """ Moves the car according to the speed (changed by action) """
        assert action in self.action_space, "Invalid action: {}".format(action)
        self.speed += self._action2move(action)
        self.speed.clip(0,4,self.speed) # speed constraints
        if not np.any(self.speed != 0): # speed can't be all 0 so add random
            self.speed[np.random.rand()<0.5] = 1
        self.position += self.speed
        self.pos_history.append(tuple(self.position))
        done = False
        # Goal checking
        if (self.position[0] in range(self.obstacle[0], self.size))\
           and (self.position[1] >= self.size-1):
            done = True
        elif self._hit_obstacle():
            self.reset()
        # Returns; anything brings a reward of -1
        return (*tuple(self.position), *tuple(self.speed)), -1, done, {}

    def render(self):
        print("Position is {}".format(self.position))
        print("Speed is {}".format(self.speed))
        s = np.zeros((self.size, self.size), dtype=int).astype(str)
        s[0, :self.obstacle[1]] = 'S'  # Start state
        s[self.obstacle[0]:, -1] = 'F'   # Finish line
        s[:self.obstacle[0], self.obstacle[1]:] = 'X'   # obstacle states
        all_pos = np.array(self.pos_history)
        s[all_pos[:,0], all_pos[:,1]] = '.'
        print(s)
