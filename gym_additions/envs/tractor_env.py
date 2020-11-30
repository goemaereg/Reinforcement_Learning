import gym
from gym import spaces
import numpy as np
import random
from .img2np import img2np
from PIL import Image

class TractorEnv(gym.Env):
    """ Environment modeling the FlandersMake parking to run a tractor there
        """
    def __init__(self):
        #self.map = img2np('~/Codes/Reinforcement_Learning/gym_additions/envs/gridworlds_img/parking_gridworld.png')
        self.map = img2np('./gridworlds_img/parking_gridworld.png')
        self.height, self.width = self.map.shape
        self.action_space = spaces.Discrete(3)
        self.poses = ['^', '>', '<', 'v'] # North East West South
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width),
                spaces.Discrete(len(self.poses))
                ))
        self.moves = { # these are orientation dependent.
                0: (-1, 0),  # front
                1: (-1, 1),   # right
                2: (-1, -1),   # left
                }
        self.noise_proba = 0. # probability to do another action (see step)
        self.terminal = np.array([3,self.width-4]) # terminal state
        self.terminal_pose = '>'
        self.start = np.array([self.height-2,1])
        self.start_pose = '>'
        # begin in start state
        self.reset()

    def state(self):
        return (*self.s, self.poses.index(self.pose))

    def reset(self):
        self.display_frames = 0
        self.s = np.array(self.start)
        self.pose = self.start_pose
        return self.state()

    def _move_from_pose(self, move):
        if self.pose is '^':
            pass # no changes to the move, this is our default orientation
        elif self.pose is 'v':
            move *= -1 # rotation by pi
        elif self.pose is '>':
            move = np.array([move[1], -move[0]]) # rotation by -pi/2
        elif self.pose is '<':
            move = np.array([-move[1], move[0]]) # rotation by pi/2

        return move

    def _pose_from_action(self, action):
        if action is 0: # front
            return self.pose # pose doesn't change
        if action is 1: # right
            if self.pose is '^':
                return '>'
            if self.pose is '>':
                return 'v'
            if self.pose is 'v':
                return '<'
            if self.pose is '<':
                return '^'
        if action is 2: # left
            if self.pose is 'v':
                return '>'
            if self.pose is '<':
                return 'v'
            if self.pose is '^':
                return '<'
            if self.pose is '>':
                return '^'

        raise ValueError("action not in (0,1,2): {}".format(action))

    def _no_obstacle_in(self, state):
        # out of bounds?
        if state[0] >= self.height or state[0] < 0 or \
           state[1] >= self.width  or state[1] < 0:
            return False
        # touched obstacle?
        if self.map[(state[0], state[1])] == 1:
            return False
        # if not, good job, you're safe
        return True

    def step_no_noise(self, action):
        """ Moves the agent in the action direction."""
        # Next, moving according to action
        move = np.array(self.moves[action])
        move_front = np.array(self.moves[0])
        move = self._move_from_pose(move)
        move_front = self._move_from_pose(move_front)
        # print("in step: state {} action {} move {}".format(self.s, action, move))

        impact = False # touched something
        done = False
        success = False
        new_s = self.s + move
        by_s = self.s + move_front # we passed by this state necessarily
        if self._no_obstacle_in(new_s) and self._no_obstacle_in(by_s):
            self.s = new_s
            self.pose = self._pose_from_action(action)
        else:
            impact = True
            done = True

        if np.alltrue(self.s == self.terminal) and self.pose is self.terminal_pose:
            done = True
            success = True

        return self.state(), int(success) - int(impact), done, {}

    def step(self, action):
        """ Adding noise """
        if np.random.rand() < self.noise_proba:
            if np.random.rand() < .5:
                action = max(action + 1, 2)
            else:
                action = min(action - 1, 0)

        return self.step_no_noise(action)


    def p(self, state, action):
        save_s = self.s
        save_pose = self.pose
        self.s = np.array([state[0], state[1]])
        self.pose = self.poses[state[2]]
        assert self._no_obstacle_in(self.s), "Dynamics: non terminal states only"
        d = {}
        next_state, reward, done, _ = self.step_no_noise(action)
        d[tuple(next_state), reward] = 1.

        self.s = save_s
        self.pose = save_pose
        return d

    def legal_actions(self, state):
        return [0,1,2]

    def non_terminal_states(self):
        """ Returns a list of all possible (non terminal) states."""
        w = np.where(self.map == 0) # non obstacle states
        positions = np.array([w[0], w[1]]).T
        return [(x,y,p) for p in range(len(self.poses)) for x,y in positions]

    def render(self, save_visuals=True):
        print_map = self.map.copy().astype(str)
        print_map[print_map == '1'] = 'X'
        print_map[print_map == '0'] = ' '
        print_map[tuple(self.terminal)] = 'T'
        print_map[tuple(self.start)] = 'S'
        print_map[tuple(self.s)] = self.pose
        print(print_map)

        self._save_visuals()


    def _save_visuals(self):
        background = Image.open("img/parking_overlay.png")
        width, height = background.size
        true_width, true_height = 23,29
        pw = width/true_width # pixel width
        ph = height/true_height # pixel height

        tractor = Image.open("img/tractor_sprite.png")
        haystack = Image.open("img/haystack.png")

        angle_per_pose = {0:0, 1:180+90, 2:90, 3:180}
        tractor = tractor.rotate(angle_per_pose[self.poses.index(self.pose)])
        tractor.thumbnail((pw*2,ph*2), Image.ANTIALIAS)
        tractor_position = (int(self.s[1]*pw), int(self.s[0]*ph))

        haystack.thumbnail((pw*2,ph*2), Image.ANTIALIAS)
        haystack_position = (int(self.terminal[1]*pw), int(self.terminal[0]*ph))

        background.paste(tractor, tractor_position , tractor)
        background.paste(haystack, haystack_position, haystack)
        filename = "img/overlay_all/{}.png".format(self.display_frames)
        background.save(filename,"PNG")
        print("Saved file as {}".format(filename))

        self.display_frames += 1
