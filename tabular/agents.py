import numpy as np
from utils import assert_not_abstract, my_argmax
from agents_core import Agent, Random_Agent
import random
from collections import deque

class MonteCarlo(Agent):
    """ Tabular method that keeps the Q-values of all the possible
    state-action pairs; updates on an episode-wise schedule """
    def __init__(self, env_shapes, epsilon=1., gamma=1., **kwargs):
        super(MonteCarlo, self).__init__(env_shapes) # inits shapes
        self.name = 'MonteCarlo'
        self.short_name = 'MC'
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
            action = my_argmax(self.Qtable[s])
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
        self.epsilon = max(self.epsilon - 2.5e-3, 0.1)
        if np.random.rand() < 0.001: print("\t\tReduced epsilon to {}".format(self.epsilon))


    def learn(self, s, a, r, s_, d=False):
        """ Only applies the update over the whole episode when the latter has
        terminated and a history is available. Stocks said history. """
        assert type(d) is bool, "MonteCarlo needs a boolean done signal."
        self.history.appendleft([s,a,r])
        if d:
            for s,a,r in self.history:
                self.G = self.G*self.gamma + r
                self.Ctable[s][a] += self.W
                old_qsa = self.Qtable[s][a]
                self.Qtable[s][a] += self.W*(self.G - self.Qtable[s][a])/self.Ctable[s][a]
                #print("Qtable update from {} to {} due to return {}".format(old_qsa, self.Qtable[s][a], self.G))
                #if a != self._greedy(s):
                    # This trajectory doesn't correspond to our policy anymore
                    # break
                # self.W /= 1 - self.epsilon + self.epsilon/self.n_actions
            self._episodic_reset()
            self._anneal_epsilon()

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={:.3f}, gamma={}".format(self.epsilon, self.gamma)


class Sarsa(Agent):
    """ Tabular method that keeps the Q-values of all the possible
    state-action pairs and acts eps-greedy """
    def __init__(self, learn_rate=0.1, gamma=0.9,
                 explo_horizon=50000, min_eps=0.1, **kwargs):
        self.name = 'Sarsa'
        self.short_name = 'Sarsa'
        super(Sarsa, self).__init__(**kwargs) # inits shapes
        print((*self.input_shape, self.n_actions))
        self.explo_horizon = explo_horizon
        self.min_eps = min_eps
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.Qtable = np.zeros((*self.input_shape, self.n_actions)) # arb init at 0
        self.verbose = False
        self.reset_eps()

    def reset_eps(self):
        self.step = 0
        self.anneal_epsilon(0)
        print("\t\t\tReset eps: {}".format(self.epsilon))

    def act(self, obs):
        """ Epsilon-greedy policy over the Qtable """
        # obs = tuple(obs) # to access somewhere in the table
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            if self.verbose:
                print("Q values of possible actions: {}".format(self.Qtable[obs]))
            action = my_argmax(self.Qtable[obs])
            return action

    def anneal_epsilon(self, ep:int):
        """ Anneals epsilon linearly based on the step number """

        self.epsilon = max((self.explo_horizon - ep)/self.explo_horizon, self.min_eps)

    def _reward_seq_discounter(self, rewards):
        """ For the option framework.
        Input rewards is a sequence of rewards [r1 r2 r3 ...]
        This function outputs the discounted sum of rewards and the discount
        for the final step (i.e. gamma**n_rewards) for the bootstrapping. """
        if np.isscalar(rewards):
            return rewards, self.gamma # no option setting, no changes.
        # else: rewards is a list (of r)
        ret = 0 # total discounted reward
        g = 1 # gamma exponentials
        for r in rewards:
            ret += r*g
            g   *= self.gamma
        return ret, g

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        Updates the annealing epsilon. """
        r, discount = self._reward_seq_discounter(r) # outputs r, gamma unless option
        self.Qtable[s][a] += self.learn_rate * (
            r + discount*self.Qtable[s_][self.act(s_)]*(1-d) - self.Qtable[s][a]
        )
        self.step += 1
        self.anneal_epsilon(self.step)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, lr={}, gamma={}"\
               .format(self.epsilon, self.learn_rate, self.gamma)


class ExpectedSarsa(Sarsa):
    """ Improvement on Sarsa to compute the expectation over all actions """
    def __init__(self, **kwargs):
        super(ExpectedSarsa, self).__init__(**kwargs) # inits shapes
        self.name = 'ExpectedSarsa'
        self.short_name = 'ESarsa'

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        ExpectedSarsa takes an expectation over possible action probailities.
        Updates the annealing epsilon. """
        r, discount = self._reward_seq_discounter(r) # outputs r, gamma unless option
        expectation = self.epsilon/self.n_actions*np.sum(self.Qtable[s_])\
                      + (1-self.epsilon)*np.max(self.Qtable[s_])
        self.Qtable[s][a] += self.learn_rate * (
            r + discount*expectation*(1-d) - self.Qtable[s][a]
        )
        self.step += 1
        self.anneal_epsilon(self.step)


class QLearning(Sarsa):
    """ Improvement on Sarsa to max Q(S',.) over all actions """
    def __init__(self, **kwargs):
        super(QLearning, self).__init__(**kwargs)
        self.name = 'QLearning'
        self.short_name = 'QL'

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        QLearning maxes over actions in the future state (off policy).
        Updates the annealing epsilon. """
        r, discount = self._reward_seq_discounter(r) # outputs r, gamma unless option (multistep)
        qa_value = self.Qtable[s][a]
        target = discount * np.max(self.Qtable[s_]) * (1-d)
        delta = r + target - qa_value
        update = self.learn_rate * delta
        self.Qtable[s][a] += update
        # self.Qtable[s][a] += self.learn_rate * (
        #     r + discount*np.max(self.Qtable[s_])*(1-d) - self.Qtable[s][a])
        self.step += 1
        self.anneal_epsilon(self.step)


class QLearning_Optimistic(QLearning):
    """ High initial Q values (default at 1) """
    def __init__(self, optimistic_value=1, **kwargs):
        self.optimistic_value = optimistic_value
        super(QLearning_Optimistic, self).__init__(**kwargs)
        self.name = 'QLearning_Optimistic'
        self.short_name = 'QLO'
        print("Using Optimistic value {}".format(self.optimistic_value))

    def reset(self):
        super(QLearning_Optimistic, self).reset()
        self.Qtable += self.optimistic_value # optimistic start


class DoubleQLearning(Sarsa):
    """ Computes two estimates Q1 and Q2 to prevent maximization bias."""

    def __init__(self, **kwargs):
        super(DoubleQLearning, self).__init__(**kwargs) # inits shapes
        self.name = 'DoubleQLearning'
        self.short_name = 'DQL'
        self.reset()

    def act(self, obs):
        """ Epsilon-greedy policy over the sum of Qtables """
        #obs = tuple(obs) # to access somewhere in the table
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
            return action
        else:
            action = my_argmax(self.Q1[obs] + self.Q2[obs])
            return action

    def learn(self, s, a, r, s_, d=False):
        """ Updates one of the Qtables based on the s,a,r,s_ transition.
        Updates the annealing epsilon. """
        r, discount = self._reward_seq_discounter(r) # outputs r, gamma unless option
        if np.random.rand() < 1/2:
            self.Q1[s][a] += self.learn_rate * (
                r + discount*self.Q2[s_][my_argmax(self.Q1[s_])]*(1-d) - self.Q1[s][a]
            )
        else:
            self.Q2[s][a] += self.learn_rate * (
                r + discount*self.Q1[s_][my_argmax(self.Q2[s_])]*(1-d) - self.Q2[s][a]
            )

        self.step += 1
        self.anneal_epsilon(self.step)

    def reset(self):
        super(DoubleQLearning, self).reset()
        self.Q1 = np.zeros((*self.input_shape, self.n_actions)) # arb init at 0
        self.Q2 = np.zeros((*self.input_shape, self.n_actions)) # arb init at 0


class DynaQ(QLearning):
    """ Model-based tabular method learning a deterministic tabular env,
    and applying QLearning to both the direct RL and planning parts. """
    def __init__(self, n, **kwargs):
        super(DynaQ, self).__init__(**kwargs) # inits shapes
        # calls reset
        self.name = 'DynaQ'
        self.short_name = 'DQ'
        self.n = n # number of steps of planning per learning step

    def reset(self):
        super(DynaQ, self).reset() # inits shapes
        self.model = {} # dictionary of (s,a) -> (r,s') transitions

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        Applies QLearning directly; improves the model; applies n QLearning
        steps on the model.
        Not adapted to options (yet?). """
        super(DynaQ, self).learn(s,a,r,s_,d)
        self.model[s,a] = r, s_, d
        for _ in range(self.n):
            (s, a), (r,s_,d) = random.choice(list(self.model.items()))
            super(DynaQ, self).learn(s,a,r,s_,d)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "n={}, eps={}, lr={}, gamma={}"\
               .format(self.n, self.epsilon, self.learn_rate, self.gamma)


class DynaQPlus(DynaQ):
    """ DynaQ with an exploration bonus.
    We do not implement (for now) the possibility to use non-modelled actions.
    """
    def __init__(self, kappa, **kwargs):
        super(DynaQPlus, self).__init__(**kwargs) # inits shapes
        # calls reset
        self.name = 'DynaQPlus'
        self.short_name = 'DQ+'
        self.kappa = kappa

    def reset(self):
        super(DynaQPlus, self).reset() # inits shapes
        self.taus = np.zeros((*self.input_shape, self.n_actions)) # non-visit counts

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        Applies QLearning directly; improves the model; applies n QLearning
        steps on the model. """
        QLearning.learn(self,s,a,r,s_,d)
        self.model[s,a] = r, s_, d
        self.taus += 1  # update all non-visit counts...
        self.taus[s][a] = 0 # except for this state-action pair
        for _ in range(self.n):
            (s, a), (r,s_,d) = random.choice(list(self.model.items()))
            explo_r = self.kappa*np.sqrt(self.taus[s][a])
            super(DynaQPlus, self).learn(s,a,r+explo_r,s_,d)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "n={}, kappa={}, eps={}, lr={}, gamma={}"\
               .format(self.n, self.kappa, self.epsilon, self.learn_rate, self.gamma)


class DynaQPlus2(DynaQPlus):
    """ DynaQPlus comparison with action-based exploration instead of reward.
    """
    def __init__(self, **kwargs):
        super(DynaQPlus2, self).__init__(**kwargs) # inits shapes
        # calls reset
        self.name = 'DynaQPlus2'
        self.short_name = 'DQ+2'

    def learn(self, s, a, r, s_, d=False):
        """ Uses the DynaQ learn method instead of DynaQPlus' which adds a
        reward boost to unvisited actions. """
        self.taus += 1  # update all non-visit counts...
        self.taus[s][a] -= 1 # except for this state-action pair
        DynaQ.learn(self,s,a,r,s_,d)

    def act(self, obs):
        """ Epsilon-greedy policy over the Qtable with a reward boost """
        # obs = tuple(obs) # to access somewhere in the table
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            if self.verbose:
                print("Q values of possible actions: {}".format(self.Qtable[obs]))
            action = my_argmax(self.Qtable[obs] + self.kappa*np.sqrt(self.taus[obs]))
            return action


class EligibilityTraces(Sarsa):
    """ Improvement on QLearning with Eligibility Traces in tabular case."""

    """ Tabular method that keeps the Q-values of all the possible
    state-action pairs and acts eps-greedy """
    def __init__(self, lbda=0.9, **kwargs):
        super(EligibilityTraces, self).__init__(**kwargs) # inits shapes
        self.name = 'EligibilityTraces'
        self.short_name = 'ElTr'
        self.lbda = lbda # lambda but it's a reserved keyword in python
        self.reset()

    def reset(self):
        self.Qtable = np.zeros((*self.input_shape, self.n_actions)) # arb init at 0
        self.traces = np.zeros_like(self.Qtable) # arb init at 0
        self.anneal_epsilon(0)
        self.step = 0
        self.verbose = False

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        Updates the annealing epsilon. """
        r, discount = self._reward_seq_discounter(r) # outputs r, gamma unless option
        self.traces *= self.lbda*discount
        self.traces[s][a] += 1 # =:replacing traces; +=: accumulating
        delta = r + discount*self.Qtable[s_][self.act(s_)]*(1-d) - self.Qtable[s][a]
        self.Qtable += self.learn_rate * delta * self.traces
        if d:
            self.traces = np.zeros_like(self.Qtable)
        self.step += 1
        self.anneal_epsilon(self.step)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, lr={}, gamma={}, lambda={}"\
               .format(self.epsilon, self.learn_rate, self.gamma, self.lbda)


class TreeBackup(EligibilityTraces):
    """ Improvement on Expected Sarsa to EligibilityTraces """
    def __init__(self, **kwargs):
        super(TreeBackup, self).__init__(**kwargs)
        self.name = 'TreeBackup'
        self.short_name = 'TB'

    def reset(self):
        super(TreeBackup, self).reset()
        self.visited_s_a = []

    def _policy(self, s):
        """ Returns the policy
        policy = (1-eps+eps/n_a for max action, eps/n_a for others)"""
        q_s = self.Qtable[s]
        policy = np.ones_like(q_s)*self.epsilon/self.n_actions
        policy[np.argmax(q_s)] += 1 - self.epsilon
        return policy

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        Updates the annealing epsilon.
        We use replacing traces.
        Following the ET for Off Policy Evaluation, we divide the traces by
        the max policy to prevent the traces from decaying too quickly.
        """
        r, discount = self._reward_seq_discounter(r) # outputs r, gamma unless option
        policy_s_a = self._policy(s)[a]
        self.traces *= self.lbda*discount*policy_s_a/(1-self.epsilon+self.epsilon/self.n_actions)
        self.traces[s][a] = 1
        expectation = self.epsilon/self.n_actions*np.sum(self.Qtable[s_])\
                      + (1-self.epsilon)*np.max(self.Qtable[s_])
        delta = r + discount*expectation*(1-d) - self.Qtable[s][a]
        self.Qtable += self.learn_rate * delta * self.traces
        if d:
            self.traces = np.zeros_like(self.Qtable)
            self.visited_s_a = []
        self.step += 1
        self.anneal_epsilon(self.step)
