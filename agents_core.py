import numpy as np
from utils import assert_not_abstract
from mdp_core import *

class Agent():
    """ An Agent is an abstract entity that can simply act in an environment.

        It might also learn given useful information.
        """

    def __init__(self, env_shapes, **kwargs):
        """ Takes in the env_shapes for input / output
            Checks abstraction by testing classname.
            """
        assert_not_abstract(self, 'Agent')
        if env_shapes is not None: # otherwise, bandits
            self.input_shape, self.n_actions = env_shapes

    def act(self, obs:State) -> Action:
        """ Acts given an observation.
            """
        raise NotImplementedError("Calling abstract method act in Agent")

    def learn(s:State, a:Action, r:float, s_:State, d=None):
        """ Learns provided usual essential information,
            i.e. transition s -> s_ given action a, resulting in reward r.
            Optional d indicates whether the state is terminal.
            Note again that s_ refers to s_{t+1} in the theory.

            Leave blank for a non-learning agent (eg random).
            """
        pass


class Random_Agent(Agent):
    """ Acts randomly in any situation.
        """
    def __init__(self, env_shapes):
        self.name = 'Random'
        super(Random_Agent, self).__init__(env_shapes) # inits shapes

    def act(self, obs):
        return np.random.choice(self.n_actions)


class QLearning(Agent):
    """ Tabular method that keeps the Q-values of all the possible
    state-action pairs """
    def __init__(self, env_shapes, epsilon=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000):
        self.name = 'QLearning'
        super(QLearning, self).__init__(env_shapes) # inits shapes
        print((*self.input_shape, self.n_actions))
        self.Qtable = np.zeros((*self.input_shape, self.n_actions)) # arb init at 0
        self.epsilon = epsilon
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.explo_horizon = explo_horizon
        self.verbose = False
        self.step = 0

    def act(self, obs):
        """ Epsilon-greedy policy over the Qtable """
        obs = tuple(obs) # to access somewhere in the table
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
            return action
        else:
            if self.verbose:
                print("Q values of possible actions: {}".format(self.Qtable[obs]))
            action = np.argmax(self.Qtable[obs])
            return action

    def anneal_epsilon(self, ep:int):
        """ Anneals epsilon linearly based on the step number """
        self.epsilon = max((self.explo_horizon - ep)/self.explo_horizon, 0.1)

    def learn(self, s, a, r, s_, d=None):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        Updates the annealing epsilon. """
        self.Qtable[s][a] += self.learn_rate * (
            r + self.gamma*np.max(self.Qtable[s_]) - self.Qtable[s][a]
        )
        self.step += 1
        self.anneal_epsilon(self.step)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, learn_rate={}, gamma={}"\
               .format(self.epsilon, self.learn_rate, self.gamma)
