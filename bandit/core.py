from utils import assert_not_abstract, str2class
import numpy as np

class Lever(object):
    """ Abstract concept of a Lever the Agent pulls.

        Any subclass of Lever should be named Name_Lever
        (if something else is necessary, )
        """
    def __init__(self):
        """ Checks abstraction.
            """
        assert self.__class__.__name__[-6:] == '_Lever', \
               "The Lever subclass name doesn't match the _Lever end convention"
        assert_not_abstract(self, 'Lever')

    def pull():
        """ Returns a stochastic reward (scalar).
            """
        raise NotImplementedError("Call of abstract method in Lever")

class Gaussian_Lever(Lever):
    """ Lever that samples its reward from a Gaussian, std=1.
        """
    def __init__(self, std=1):
        super(Gaussian_Lever, self).__init__()
        self.mean = np.random.randn()
        self.std = std

    def pull(self):
        """ Samples from a standardized gaussian with mean self.mean.
            """
        return self.mean + np.random.randn() * self.std

class NonStat_Gaussian_Lever(Gaussian_Lever):
    """ Gaussian_Lever which randomly walks its mean in some direction every step."""

    def __init__(self, mean_std=.01):
        super(NonStat_Gaussian_Lever, self).__init__()
        self.mean_std = mean_std

    def pull(self):
        """ Moves its mean with a Gaussian (0,mean_std) before sampling
            as would a Gaussian_Lever.
            """
        self.mean += np.random.randn() * self.mean_std
        return Gaussian_Lever.pull(self) # probably clearer than using super

class Bandit(object):
    """ Set of Levers which the Agent pulls to get Rewards.
        """

    def __init__(self, lever_types):
        """ Creates the list of Levers given a list of lever types.
            New lever types may be added to the list in instantiate_lever,
            provided these exist in the levers above.
            """
        self.n_levers = len(lever_types) # or "arm"
        self.levers = [None] * self.n_levers
        self.lever_types = lever_types
        self.reset()

    def reset(self):
        """ Resets (new values for the levers) """
        for i, lever_type in enumerate(self.lever_types):
            self.levers[i] = lever_type()

    def pull(self, l):
        """ Returns the reward for pulling the lever l. """
        return self.levers[l].pull()

    def reveal_qstar(self):
        """ Reveals the true q_* values of the actions """
        return np.array([lever.mean for lever in self.levers])
