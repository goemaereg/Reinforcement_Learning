import numpy as np
from utils import assert_not_abstract
from mdp_core import *
from collections import deque

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
        super(Random_Agent, self).__init__(env_shapes) # inits shapes
        self.name = 'Random'

    def act(self, obs):
        return np.random.choice(self.n_actions)


class ValueIteration(Agent):
    """ Dynamic Programming Agent that performs ValueIteration.
    The learn method is overwritten to only take in a state.
    The act method should be used after multiple sweeps over the state space
    (using a learn method for each and every state of the space).
    The resulting value function should be optimal provided enough time."""

    def __init__(self, env):
        self.env = env # environment we're trying to solve
        self.env_shapes = (env.observation_space.n, env.action_space.n)
        super(ValueIteration, self).__init__(self.env_shapes)
        self.name = "ValueIteration"
        self.V = np.random.randn(self.input_shape)*0.
        self.V[self.env.terminal_states()] = 0
        self.gamma = 1.

    def learn(self, s:State):
        """ Value Iteration update """
        actions = self.env.legal_actions(s)
        ps = {a:self.env.p(s,a) for a in actions} # p(s',r | s,.) as dictionaries
        # print("State {} with dictionaries {}".format(s, ps))
        value_array = [sum([ps[a][s_,r]*(r+self.gamma*self.V[s_])
                            for s_,r in ps[a]])
                       for a in actions]
        self.V[s] = max(value_array)

    def act(self, s):
        actions = self.env.legal_actions(s)
        ps = {a:self.env.p(s,a) for a in actions} # p(s',r | s,.) as dictionaries
        value_array = [sum([ps[a][s_,r]*(r+self.gamma*self.V[s_])
                            for s_,r in ps[a]])
                       for a in actions]
        return actions[np.argmax(value_array)]


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


class MonteCarlo(Agent):
    """ Tabular method that keeps the Q-values of all the possible
    state-action pairs; updates on an episode-wise schedule
    This version is off-policy, with an epsilon greedy exploring and
    learning the greedy. """
    def __init__(self, env_shapes, epsilon=1., gamma=1.):
        super(MonteCarlo, self).__init__(env_shapes) # inits shapes
        self.name = 'MonteCarlo'
        print((*self.input_shape, self.n_actions))
        self.gamma = gamma
        self.epsilon = epsilon
        self.verbose = False
        self.reset()

    def _episodic_reset(self):
        """ Resets the history, G and W variables for another episode """
        self.history = deque()  # so we don't have to invert the queue
        self.G = 0
        self.W = 1

    def reset(self):
        self.Qtable = np.zeros((*self.input_shape, self.n_actions)) # arb init at 0
        self.Ctable = np.zeros((*self.input_shape, self.n_actions)) # arb init at 0
        self._episodic_reset()

    def _greedy(self, s):
        """ Acts greedy w.r.t our Qtable."""
        try:
            action = np.argmax(self.Qtable[s])
        except Exception as e:
            print(e, "\nInvalid state {} for input_shape {}".format(s, self.input_shape))
            quit()
        return action

    def act(self, s):
        """ Epsilon-greedy policy over the Qtable """
        s = tuple(s) # to access somewhere in the table
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = self._greedy(s)

        return action

    def _anneal_epsilon(self):
        self.epsilon = max(self.epsilon - 1e-3, 0.1)
        if np.random.rand() < 0.01: print("\t\tReduced epsilon to {}".format(self.epsilon))

    def learn(self, s, a, r, s_, d=None):
        """ Only applies the update over the whole episode when the latter has
        terminated and a history is available. Stocks said history. """
        assert type(d) is bool, "MonteCarlo needs a boolean done signal."
        if d:
            for s,a,r in self.history:
                self.G = self.G*self.gamma + r
                self.Ctable[s][a] += self.W
                self.Qtable[s][a] += self.W*(self.G - self.Qtable[s][a])/self.Ctable[s][a]
                if a != self._greedy(s):
                    # This trajectory doesn't correspond to our policy anymore
                    break
                self.W *= 1 - self.epsilon + self.epsilon/self.n_actions
            self._episodic_reset()
            self._anneal_epsilon()
        else:
            self.history.appendleft([s,a,r])

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, gamma={}"\
               .format(self.epsilon, self.gamma)
