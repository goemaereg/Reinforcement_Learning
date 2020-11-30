import gym
from gym import spaces
import numpy as np
import random

class FlappyBirdEnv(gym.Env):
    """ Flappy Bird game with two inputs (x,y position diff to the obstacle)
    Two actions - to flap or not to flap. """
    def __init__(self, discretized=True):
        # Input Outputs
        self.width = 10 # Discretized space (only at state output)
        self.height = 10
        self.width_height = np.array([self.width, self.height])
        self.action_space = spaces.Discrete(2) # Flap (1) or not (0)
        self.observation_space = spaces.Tuple(( # Discretized input
                spaces.Discrete(self.width), # x axis, width. [-1, 1]
                spaces.Discrete(self.height) # y axis, height. [-1, 1]
                ))

        # Init world
        self.g = 9.8 # gravity
        self.dt = .1 # TO BE TOYED WITH
        self.living_reward = .1 # Every frame the bird is alive

        # Init bird properties
        self.bird_radius = .05 # Diameter is .1 - Bird is actually a square
        self.bird_yspeed = np.random.randn()
        flap_height = .3 # !! To be toyed with
        self.flap_v = np.sqrt(2*self.g*flap_height) # Flap velocity boost

        # Init obstacle properties
        self.obst_width = .2 # !! To be toyed with
        self.obst_space = .3 # space between top and bottom parts !! To be toyed with
        self.obst_xspeed = 1 # !! To be toyed with

        # Init positions
        self.reset()

    def _discretize(self, pos):
        """ Computes the discretized version of the pos.
        pos has to be a numpy array in [-1, 1]
        Output is in ({0..width}, {0..height})"""
        # rescale the pos to [0,1] then apply round function
        return np.floor((pos+1)/2 * self.width_height).astype(int)

    def reset(self):
        """ Sets initial positions to the bird and obstacles """
        # Bird position
        randy = np.random.uniform(2*self.bird_radius, 1-2*self.bird_radius)
        self.bird_pos = np.array([0, rand_y])

        # Obstacle position -- the hitbox is defined from just x,y
        randx = np.random.uniform(.6, 1)
        randy = np.random.uniform(.2, .8)
        self.obst_pos = np.array([randx, randy])

        return self._discretize(self.bird_pos - self.obst_pos)

    def _check_collision(self):
        """ Collision can be with the border of the map or the obstacle """
        # Bird hits borders of the map (only y values)
        if (self.bird_pos[1] < 0) or (self.bird_pos[1] > 1):
            return True

        # Lower
        dist = np.abs(self.bird_pos - self.obst_pos) # norm 1 distance
        if  (dist[0] < bird.radius + self.obst_width/2) \
        and (dist[1] > bird.radius + self.obst_space/2):
            return True

        return False


    def step(self, action):
        """ Applies physics to the bird. v += a*dt
        Flap applies the flap boost, v += flap_v.
        Checks collision. """
        assert action in self.action_space, "Invalid action: {} not in {}".format(action, self.action_space)

        # Apply physics to the bird
        self.bird_yspeed += -self.g*self.dt + self.flap_v*action
        seld.bird_pos[1] += self.bird_yspeed*self.dt

        # Move obstacle
        self.obst_pos[0] += self.obst_xspeed*self.dt

        # Check collision
        done = self._check_collision()

        state = self._discretize(self.bird_pos - self.obst_pos)
        return state, self.living_reward, done, {}

    def render(self):

        pass
