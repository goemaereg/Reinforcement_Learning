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

    def reset(self):
        """ Resets the agent to tabula rasa, it hasn't learnt anything yet."""
        pass

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
    def __init__(self, env_shapes, **kwargs):
        super(Random_Agent, self).__init__(env_shapes) # inits shapes
        self.name = 'Random_Agent'
        self.short_name = 'Rand'

    def act(self, obs):
        return np.random.choice(self.n_actions)


class ValueIteration_MRP(Agent):
    """ Dynamic Programming Agent that performs ValueIteration.
    The learn method is overwritten to only take in a state.
    This class is for Markov Reward Processes, hence doesn't have actions.
    The resulting value function should be optimal provided enough time."""

    def __init__(self, env):
        self.env = env # environment we're trying to solve
        self.env_shapes = (env.observation_space.n, None)
        super(ValueIteration_MRP, self).__init__(self.env_shapes)
        self.name = "ValueIteration_MRP"
        self.V = np.random.randn(self.input_shape)*0.
        self.V[self.env.terminal_states()] = 0
        self.gamma = 1.

    def learn(self, s:State):
        """ Value Iteration update """
        p_s = self.env.p(s)
#        print("Learning from state {} with dic {}".format(s, p_s))
#        print("Old value of Vs: {}".format(self.V[s]))
        self.V[s] = sum([p_s[s_,r]*(r+self.gamma*self.V[s_]) for s_,r in p_s])
#        print("New value of Vs: {}".format(self.V[s]))


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


class TD0(object):
    """ Simple TD(0) algorithm."""

    def __init__(self, env_shapes, gamma=0.9, learn_rate=0.1):
        self.input_shape, self.n_actions = env_shapes
        self.name = "ValueIteration"
        self.V = np.zeros(self.input_shape)
        self.gamma = gamma
        self.learn_rate = learn_rate

    def learn(self, s, r, s_, d):
        """ TD(0) update """
        self.V[s] += self.learn_rate*(
            r + self.gamma*self.V[s_]*(1-d) - self.V[s]
        )
