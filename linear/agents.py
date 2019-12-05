import numpy as np
from utils import assert_not_abstract, my_argmax, my_random_choice, prod, softmax
from agents_core import Agent
import random
from collections import deque
import warnings
#warnings.filterwarnings("error")

class Sarsa(Agent):
    """ Linear method that act eps-greedy and uses TD(0) as update """
    def __init__(self, env_shapes, min_eps=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(Sarsa, self).__init__(env_shapes) # inits shapes
        self.name = 'Sarsa'
        self.input_dim = prod(self.input_shape) # flattens the shape
        print((self.input_dim, self.n_actions))
        self.explo_horizon = explo_horizon
        self.min_eps = min_eps
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.w = np.zeros((self.input_dim, self.n_actions)) # arb init at 0
        self.anneal_epsilon(0)
        self.step = 0
        self.verbose = False

    def _q_hat(self, x, a=None):
        """ Q function approximator - in this case, linear.
        If a is None (or not provided), the whole table for all a is returned.
        """
        q_x = np.dot(self.w.T,x)
        return q_x if a is None else q_x[a]

    def act(self, x):
        """ Epsilon-greedy policy over w """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            action_values = self._q_hat(x) # definition of linear q_hat
            if np.random.rand()<0.000001: # and self.verbose:
                print("Q values of possible actions: {}".format(action_values))
            return my_argmax(action_values)

    def anneal_epsilon(self, ep:int):
        """ Anneals epsilon linearly based on the step number """
        self.epsilon = max((self.explo_horizon - ep)/self.explo_horizon, self.min_eps)

    def learn(self, x, a, r, x_, d=None):
        """ Updates the weight vector based on the x,a,r,x_ transition,
        where x is the feature vector of current state s.
        Updates the annealing epsilon. """
        target = self._q_hat(x_, self.act(x_))
        delta = r + self.gamma*target - self._q_hat(x, a)
        if np.random.rand() <0.0001: print("Learning: delta: {} + {}*{} - {} = {}".format(r,self.gamma, target, self._q_hat(x, a), delta))
        self.w[:,a] += self.learn_rate*delta*x
        # anneal epsilon
        self.step += 1
        self.anneal_epsilon(self.step)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, learn_rate={}, gamma={}"\
               .format(self.min_eps, self.learn_rate, self.gamma)


class ExpectedSarsa(Sarsa):
    """ Improvement on Sarsa to compute the expectation over all actions """
    def __init__(self, env_shapes, min_eps=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(ExpectedSarsa, self).__init__(env_shapes, min_eps, learn_rate, gamma, explo_horizon) # inits shapes
        self.name = 'ExpectedSarsa'

    def learn(self, x, a, r, x_, d=None):
        """ Updates the w based on the x,a,r,x_ transition.
        ExpectedSarsa takes an expectation over possible action probabilities.
        Updates the annealing epsilon. """
        q_hat_x_ = self._q_hat(x_)
        expectation = self.epsilon/self.n_actions*np.sum(q_hat_x_)\
                      + (1-self.epsilon)*np.max(q_hat_x_)
        delta = r + self.gamma*expectation - self._q_hat(x, a)
        self.w[:,a] += self.learn_rate*delta*x
        # anneal epsilon
        self.step += 1
        self.anneal_epsilon(self.step)

class QLearning(Sarsa):
    """ Improvement on Sarsa to max Q(S',.) over all actions """
    def __init__(self, env_shapes, min_eps=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(QLearning, self).__init__(env_shapes, min_eps, learn_rate, gamma, explo_horizon) # inits shapes
        self.name = 'QLearning'

    def learn(self, x, a, r, x_, d=None):
        """ Updates the w based on the x,a,r,x_ transition.
        QLearning maxes over actions in the future state (off policy).
        Updates the annealing epsilon. """
        delta = r + self.gamma*max(self._q_hat(x_)) - self._q_hat(x, a)
        self.w[:,a] += self.learn_rate*delta*x
        self.step += 1
        self.anneal_epsilon(self.step)

class SarsaLambda(Sarsa):
    """ Linear Sarsa(lambda) - i.e. with Eligibility Traces"""

    def __init__(self, env_shapes, lmbda=0.9, epsilon=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(SarsaLambda, self).__init__(env_shapes, epsilon, learn_rate, gamma, explo_horizon) # inits shapes
        self.name = 'SarsaLambda'
        self.lmbda = lmbda # lambda but it's a reserved keyword in python
        self.reset()

    def reset(self):
        super(SarsaLambda, self).reset()
        self.z = np.zeros_like(self.w)

    def learn(self, x, a, r, x_, d=None):
        """ Updates the q weights based on the x,a,r,x_ transition.
        Updates the annealing epsilon. """
        grad = np.zeros_like(self.w)
        grad[:,a] = x
        self.z = self.z*self.lmbda*self.gamma + grad
        delta = r + self.gamma*self._q_hat(x_, self.act(x_)) - self._q_hat(x,a)
        #print("delta: r + gamma*maxqx_ - qx = {} + {}*{} - {} = {}".format(r,self.gamma, max(self._q_hat(x_)), self._q_hat(x,a), delta))
        self.w += self.learn_rate*delta*self.z
        if d:
            self.z = np.zeros_like(self.w)
        self.step += 1
        self.anneal_epsilon(self.step)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, learn_rate={}, gamma={}, lmbda={}"\
               .format(self.min_eps, self.learn_rate, self.gamma, self.lmbda)

class TrueSarsaLambda(Sarsa):
    """ True Online Sarsa(lambda) as in presented in the RL book. """

    def __init__(self, env_shapes, lmbda=0.9, epsilon=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(TrueSarsaLambda, self).__init__(env_shapes, epsilon, learn_rate, gamma, explo_horizon) # inits shapes
        self.name = 'TrueSarsaLambda'
        self.lmbda = lmbda # lambda but it's a reserved keyword in python
        self.reset()

    def reset(self):
        super(TrueSarsaLambda, self).reset()
        self.z = np.zeros_like(self.w)
        self.q_old = 0

    def learn(self, x, a, r, x_, d=None):
        """ Updates the q weights based on the x,a,r,x_ transition.
        Updates the annealing epsilon. """
        grad = np.zeros_like(self.w)
        grad[:,a] = x
        q  = self._q_hat(x, a)
        q_ = self._q_hat(x_, self.act(x_))
        target = r + self.gamma*q_
        self.z = self.lmbda*self.gamma*self.z + \
                 (1 - self.learn_rate*self.lmbda*self.gamma*np.dot(self.z[:,a].T, x))*grad
        #print("delta: r + gamma*maxqx_ - qx = {} + {}*{} - {} = {}".format(r,self.gamma, max(self._q_hat(x_)), self._q_hat(x,a), delta))
        self.w += self.learn_rate*( (target - self.q_old)*self.z\
                                    - (q - self.q_old)*grad )
        self.q_old = q_
        if d:
            self.z = np.zeros_like(self.w)
        self.step += 1
        self.anneal_epsilon(self.step)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, learn_rate={}, gamma={}, lmbda={}"\
               .format(self.min_eps, self.learn_rate, self.gamma, self.lmbda)

class Reinforce(Agent):
    """ Monte Carlo Reinforce Algorithm"""

    def __init__(self, env_shapes, gamma, learn_rate=2.5e-4, temperature=1, **kwargs):
        super(Reinforce, self).__init__(env_shapes) # inits shapes
        self.name = 'Reinforce'
        self.input_dim = prod(self.input_shape) # flattens the shape
        print((self.input_dim, self.n_actions))
        self.learn_rate = learn_rate
        self.temperature = temperature
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.theta = np.random.randn(self.input_dim, self.n_actions) # arb init at 0
        self.verbose = False
        self.history = deque()
        self.grad_logs = deque() # log gradients per steps in episode

    def _h(self, x, a=None):
        """ Returns the preference for action a in state with features x.
        If a is not provided, the whole array is returned. """
        h_x = np.dot(self.theta.T, x)
        return h_x if a is None else h_x[a]

    def _policy(self, x):
        return softmax(self._h(x), self.temperature)

    def act(self, x):
        """ Selects an action sampling the softmax wrt preferences h=w.T@x"""
        return my_random_choice(range(self.n_actions), self._policy(x))

    def learn(self, x, a, r, x_, d=None):
        """ Monte Carlo training of Reinforce.
        Unlike in the book, the update targets should be computed without
        changing the value of theta, otherwise the policy changes.
        To achieve this, we can either stack updates, or compute them online
        and record them. We use the latter, although it requires memory, since
        it evens out the computational requirements over the episode.
        """
        self.history.appendleft([x,a,r])
        policy = self._policy(x)
        grad = np.zeros_like(self.theta)
        grad[:,a] = x
        grad -= np.array([pi_a*x for pi_a in policy]).T
        self.grad_logs.appendleft(grad)
        if d: # MC policy gradient step for all steps of the episode
            G = 0
            T = len(self.history)
            gam_t = self.gamma**(T-1)
            for grad_log,(x,a,r) in zip(self.grad_logs, self.history):
                G = r + G*self.gamma
                self.theta += self.learn_rate*gam_t*G*grad_log
                gam_t /= self.gamma
            self.history = deque()
            self.grad_logs = deque()

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "learn_rate={}, gamma={}, temp={}"\
               .format(self.learn_rate, self.gamma, self.temperature)

class ReinforceBaseline(Reinforce):
    """ Reinforce with a Baseline from a value function."""

    def __init__(self, env_shapes, gamma, learn_rate=2.5e-4, learn_rate_w=0.01, temperature=1, **kwargs):
        self.input_dim_theta, self.input_dim_w = env_shapes[0]
        print(self.input_dim_theta, self.input_dim_w)
        self.n_actions = env_shapes[1]
        self.learn_rate = learn_rate
        self.learn_rate_w = learn_rate_w
        self.temperature = temperature
        self.gamma = gamma
        self.name = 'ReinforceBaseline'
        self.reset()

    def reset(self):
        self.theta = np.zeros((self.input_dim_theta, self.n_actions)) # arb init at 0
        self.w = np.zeros(self.input_dim_w) # arb init at 0
        self.verbose = False
        self.history = deque()
        self.grad_logs = deque() # log gradients per steps in episode

    def act(self, x):
        """ Selects an action sampling the softmax wrt preferences h=w.T@x"""
        x_theta, x_w = x
        return my_random_choice(range(self.n_actions), self._policy(x_theta))

    def _v(self, x_w):
        """ Value function (baseline) """
        return np.dot(self.w.T, x_w)

    def learn(self, x, a, r, x_, d=None):
        """ Monte Carlo on Reinforce and value function """
        self.history.appendleft([x,a,r])
        x_theta, x_w = x
        policy = self._policy(x_theta)
        grad = np.zeros_like(self.theta)
        grad[:,a] = x_theta
        grad -= np.array([pi_a*x_theta for pi_a in policy]).T
        self.grad_logs.appendleft(grad)
        if np.random.rand()<0.0001: print("Value: {}".format(self._v(x_w)))
        if d: # MC policy gradient step for all steps of the episode
            G = 0
            T = len(self.history)
            gam_t = self.gamma**(T-1)
            updates_theta = np.zeros_like(self.theta)
            updates_w = np.zeros_like(self.w)
            for grad_log,(x,a,r) in zip(self.grad_logs, self.history):
                x_theta, x_a = x
                G = r + G*self.gamma
                delta = G - self._v(x_w)
                updates_w += delta*x_w
                updates_theta += gam_t*delta*grad_log
                gam_t /= self.gamma
            self.w += self.learn_rate_w*updates_w
            self.theta += self.learn_rate*updates_theta
            self.history = deque()
            self.grad_logs = deque()
            self.v_xs = deque() # stores v estimates of states during ep

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "learn_rate={}, learn_rate_w={}, gamma={}, temp={}"\
               .format(self.learn_rate, self.learn_rate_w, self.gamma, self.temperature)

class ReinforceTileBaseline(ReinforceBaseline):
    """ Reinforce with a Baseline from a value function, optimized for Tiles."""

    def __init__(self, env_shapes, gamma, learn_rate=2.5e-4, learn_rate_w=0.01, temperature=1, **kwargs):
        super(ReinforceTileBaseline, self).__init__(env_shapes, gamma, learn_rate, learn_rate_w, temperature)
        self.v_xs = deque() # log gradients per steps in episode

    def reset(self):
        super(ReinforceTileBaseline, self).reset()
        self.v_xs = deque() # log gradients per steps in episode

    def act(self, x):
        """ Selects an action sampling the softmax wrt preferences h=w.T@x"""
        x_theta, x_w = x
        return my_random_choice(range(self.n_actions), self._policy(x_theta))

    def _v(self, x_w_tile):
        """ Value function (baseline) for Tile Coding """
        return sum(self.w[x_w_tile])

    def learn(self, x, a, r, x_, d=None):
        """ Monte Carlo on Reinforce and value function """
        self.history.appendleft([x,a,r])
        x_theta, x_w_tile = x
        policy = self._policy(x_theta)
        grad = np.zeros_like(self.theta)
        grad[:,a] = x_theta
        grad -= np.array([pi_a*x_theta for pi_a in policy]).T
        self.grad_logs.appendleft(grad)
        self.v_xs.appendleft(self._v(x_w_tile))
        if np.random.rand()<0.0001: print("Value: {}".format(self._v(x_w_tile)))
        if d: # MC policy gradient step for all steps of the episode
            G = 0
            T = len(self.history)
            gam_t = self.gamma**(T-1)
            updates_theta = np.zeros_like(self.theta)
            updates_w = np.zeros_like(self.w)
            for grad_log, v_x, (x,a,r) in zip(self.grad_logs, self.v_xs, self.history):
                x_theta, x_a = x
                G = r + G*self.gamma
                delta = G - v_x
                self.w[x_w_tile] += self.learn_rate_w*delta
                self.theta += self.learn_rate*gam_t*delta*grad_log
                gam_t /= self.gamma
            self.history = deque()
            self.grad_logs = deque()
            self.v_xs = deque()

class ActorCriticLambda(ReinforceBaseline):
    """ Using the value function not only as a Baseline but also for bootstrap.
    Naturally extended to EligibilityTraces."""

    def __init__(self, env_shapes, gamma, learn_rate=2.5e-4, learn_rate_w=0.01,
                 temperature=1, lmbda_theta=0.9, lmbda_w=0.9, **kwargs):
        super(ActorCriticLambda, self).__init__(env_shapes, gamma, learn_rate, learn_rate_w, temperature)
        self.lmbda_theta = lmbda_theta
        self.lmbda_w = lmbda_w
        self.name = 'ActorCriticLambda'

    def reset(self):
        self.theta = np.zeros((self.input_dim_theta, self.n_actions)) # arb init at 0
        self.w = np.zeros(self.input_dim_w) # arb init at 0
        self.verbose = False
        self.z_theta = np.zeros_like(self.theta)
        self.z_w = np.zeros_like(self.w)
        self.gam_t = 1

    def learn(self, x, a, r, x_, d=None):
        """ Actor Critic with EligibilityTraces """
        x_theta, x_w = x
        x_theta_, x_w_ = x_
        if np.random.rand()<0.0001: print("Value: {}".format(self._v(x_w)))

        delta = r + self.gamma*self._v(x_w_)*(1-d) - self._v(x_w)
        self.z_w = self.lmbda_theta*self.gamma*self.z_w + x_w
        grad_log = np.zeros_like(self.theta)
        grad_log[:,a] = x_theta
        policy = self._policy(x_theta)
        grad_log -= np.array([pi_a*x_theta for pi_a in policy]).T
        self.z_theta = self.lmbda_theta*self.gamma*self.z_theta \
                       + self.gam_t*grad_log
        self.w += self.learn_rate_w*delta*self.z_w
        self.theta += self.learn_rate*delta*self.z_theta

        self.gam_t *= self.gamma
        if d:
            self.z_theta = np.zeros_like(self.theta)
            self.z_w = np.zeros_like(self.w)
            self.gam_t = 1

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "lr={}, lr_w={}, gamma={}, T={}, l={}, l_w={}"\
               .format(self.learn_rate, self.learn_rate_w, self.gamma,
                       self.temperature, self.lmbda_theta, self.lmbda_w)
