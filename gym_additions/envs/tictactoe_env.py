import gym
from gym import spaces
import numpy as np


class TicTacToeEnv(gym.Env):
    """ Simple TicTacToe.
        Players play in turns and attempt to form a line.
        """
    def __init__(self):
        """ Sets the initial state board and players.
            We use a simple np array for the board.
            """
        self.player = 1
        self.opponent = -1
        self.empty = 0
        self.trans = {self.player:  [1,0],
                      self.opponent:[0,1],
                      self.empty:   [0,0]} # transformation dict for nn input
        self.width = 3
        self.height = 3
        self.action_space = spaces.Discrete(self.height*self.width)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))

    def switch_player(self):
        """ Player becomes opp, and reverse.
            """
        self.player, self.opponent = self.opponent, self.player

    def move(self,move):
        """ Checks the validity of the move first.
            Current 'player' performs the move, and it is the opponent to move.
            """
        assert move in self.legal_moves(), "Non legal move :" + str(move)

        self.fields.ravel()[move] = self.player
        self.switch_player()

    def legal_moves(self):
        """ Returns the set of legal moves in this state.
            Empty if none.
            """
        return np.flatnonzero(self.fields.ravel() == self.empty)

    def random_move(self):
        """ Returns a random move in this position.
            The move is NOT played.
            """
        return np.random.choice(self.legal_moves())

    def render(self):
        """ Prints the board (a numpy array)
            """
        print(self.fields)

    def reset(self):
        """ Gym-like method : resets the board to emptiness,
            returns the state.
            """
        self.fields = np.zeros((self.height, self.width), np.int8)
        return self.fields

    def copy_other(self, other):
        """ Gives this instance the value of another TicTacToeEnv instance.
            """
        assert isinstance(other, TicTacToeEnv), "can't copy a non-TicTacToeEnv."
        self.fields = other.fields.copy()
        self.player, self.opponent = other.player, other.opponent

    def make_copy(self):
        """ Returns a copy of this TicTacToeEnv instance.
            """
        c = TicTacToeEnv()
        c.fields = self.fields.copy()
        c.player, c.opponent = self.player, self.opponent
        return c

    def get_fields(self):
        """ Returns a copy of this instance's fields.
            """
        return self.fields.copy()

    def won(self):
        """ Returns whether the last move was winning.
            Checks the opponent's pieces since they moved last.

            Probably not the fastest solution.
            """
        if any(np.sum(self.fields, axis=0) == 3*self.opponent):
            return True
        if any(np.sum(self.fields, axis=1) == 3*self.opponent):
            return True
        if np.trace(self.fields) == 3*self.opponent:
            return True
        if ((self.fields[0,2] + \
             self.fields[1,1] + \
             self.fields[2,0]) == 3*self.opponent):
            return True
        return False

    def step(self, action):
        """ Gym-like method for a move in the environment.
            Point is to return the reward (-1,0,1) as the game winner.
            Also returns the board and a boolean for game ending.
            """
        self.move(action)
        obs = self.fields.reshape((self.height, self.width))
        done = self.won()
        reward = self.opponent if done else 0 # already switched
        if not any(self.legal_moves()) and not done:
            done = True
            reward = 0
        info = {}
        return obs, reward, done, info
