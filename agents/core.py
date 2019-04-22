import numpy as np
from utils import assert_not_abstract

class Agent():
    """ An Agent is an abstract entity that can simply act in an environment.

        It can also be able to learn given useful information.
        """

    def __init__(self, env_shapes, **kwargs):
        """ Takes in the env_shapes for input / output.

            Checks abstraction by testing classname.
            """
        assert_not_abstract(self, 'Agent')
        self.input_shape, self.n_actions = env_shapes

    def act(self, obs):
        """ Acts given an observation.
            """
        raise NotImplementedError("Calling abstract method act in Agent")

    def learn(s, a, r, s_):
        """ Learns provided usual essential information,
            i.e. transition s - s_ given action a, resulting in reward r.

            Leave blank for a non-learning agent.
            """
        pass


class Random_Agent(Agent):
    """ Acts randomly in any situation.
        """
    def act(self, obs):
        return np.random.choice(self.n_actions)
