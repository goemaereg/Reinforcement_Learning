import numpy as np
from utils import *
from agents_core import Agent, Random_Agent
from .agents import *
from .intrinsic_reward_functions import *
import random
from collections import deque


class ExploreOption(Agent):
    """ ExploreOption regroups a Meta agent having access to atomic actions + an
    exploration action; and a Explorer, triggered by this last action, that
    is trained on intrinsic rewards motivating exploration.
    """

    def __init__(self, exploiter_class, explorer_class, lrEO=2.5e-4,
                 gamma_explo=0.9, min_eps_explo=0.25, c_switch=10, beta=1,
                 reward_function=Negative_sqrt, **kwargs):
        ## Name init
        self.name = 'ExploreOption'
        self.short_name = 'XOpt'

        ## Shapes init
        super(ExploreOption, self).__init__(**kwargs) # inits shapes
        env_shapes_meta = (self.input_shape, self.n_actions+1)
        env_shapes_explo = (self.input_shape, self.n_actions)

        ## Miscellaneous hyperparameters.
        self.lrEO = lrEO # learn rate Explore Option
        self.c_switch = c_switch
        kwargs['explo_horizon'] = 1 # no annealing needed in ExploreOption
        self.learn_rate = kwargs['learn_rate']
        self.gamma = kwargs['gamma']
        self.beta = beta # fundamentally USELESS parameter to scale the Explorer's Q values.
        self.gamma_explo = gamma_explo
        self.reward_function = reward_function(env_shapes_explo) # instanciate function

        ## Initializing Exploiter
        kwargs['env_shapes'] = env_shapes_meta
        self.exploiter = exploiter_class(**kwargs)

        ## Initializing Explorer
        kwargs['env_shapes'] = env_shapes_explo
        kwargs['min_eps'] = min_eps_explo
        kwargs['gamma'] = self.gamma_explo
        if explorer_class is QLearning_Optimistic:
            kwargs['optimistic_value'] = self.beta/(1-self.gamma_explo)
        self.explorer = explorer_class(**kwargs)
        ## Variables initialization
        self.test_mode = False # i.e. we're in training mode by default
        self.reset()

    def epsilon():
        doc = "The epsilon property."
        def fget(self):
            return self.exploiter.epsilon
        def fset(self, value):
            self._epsilon = value
            self.exploiter.epsilon = value
        def fdel(self):
            del self._epsilon
        return locals()
    epsilon = property(**epsilon())

    def min_eps():
        doc = "The min_eps property."
        def fget(self):
            return self._min_eps
        def fset(self, value):
            self._min_eps = value
            self.exploiter.min_eps = value
        def fdel(self):
            del self._min_eps
        return locals()
    min_eps = property(**min_eps())

    def min_eps_explo():
        doc = "The min_eps_explo property."
        def fget(self):
            return self._min_eps_explo
        def fset(self, value):
            self._min_eps_explo = value
            self.explorer.min_eps_explo = value
        def fdel(self):
            del self._min_eps_explo
        return locals()
    min_eps_explo = property(**min_eps_explo())

    def reset(self):
        self.step = 0
        self.verbose = False
        self.exploring = 0 # explorer is acting (goes up to c_switch)
        self.explo_rewards = [] # list of rewards during exploration (for option training)
        self.exploiter.reset()
        self.explorer.reset()
        self.visit_counts = np.zeros(self.input_shape)
        self.reward_function.reset()

    def _act_no_option(self, obs):
        """ Epsilon greedy on the primitive actions. """
        if np.random.rand() < self.exploiter.epsilon:
            return np.random.randint(self.n_actions)
        else:
            action = my_argmax(self.exploiter.Qtable[obs][:-1]) # no explore option
            return action

    def act(self, obs):
        """ If the Explorer isn't on, the Meta acts. If it chooses the explore
        action, the Explorer takes over for the next c_switch steps """
        # obs = tuple(obs) # to access somewhere in the table
        if self.test_mode:
            action = self._act_no_option(obs)
            return action

        if self.exploring == 0: # meta acts
            if np.random.rand() <0.00001: print("\tMeta Q values: {}".format(self.exploiter.Qtable[obs]))
            action = self.exploiter.act(obs)
            self.explored = False

            if action == self.n_actions: # last action = explore
                self.obs_in_explo = obs # observation from which we explored
                #print("EO in state {}".format(self.obs_in_explo))
                self.exploring = self.c_switch

        if self.exploring > 0: # not 'else' since 'exploring' changes in if
            action = self.explorer.act(obs)
            self.explored = True # last action is an exploration
            self.exploring -= 1

        return action

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_transition.
        Updates the annealing epsilon.
        Here the reward r has to be a scalar """
        ## outputs r, gamma unless option:
        r, discount = self.exploiter._reward_seq_discounter(r)

        ## Training both agents on- or off-policy
        self.exploiter.learn(s,a,r,s_,d)
        self.visit_counts[s_] += 1 # For display only now
        explo_r = self.reward_function.give_reward(s,a,s_) # intrinsic reward
        if np.random.rand() < 0.00001:
            print("received explo reward {}".format(explo_r))
        self.explorer.learn(s, a, self.beta*explo_r, s_) # no access to d

        ## Training the explore option
        if d: # stop exploring and update if the episode terminates
            self.exploring = 0

        if self.explored:
            self.explo_rewards += [r]
            if self.exploring == 0: # finished exploration on last action

                self.exploiter.learn_rate = self.lrEO
                self.exploiter.learn(self.obs_in_explo, self.n_actions, self.explo_rewards, s_, d)
                self.exploiter.learn_rate = self.learn_rate
                self.explo_rewards = []

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "lr={}, lrEO={}, gam={}, c_switch={}, agts:{}&{}"\
               .format(self.learn_rate, self.lrEO, self.gamma, self.c_switch, self.exploiter.short_name, self.explorer.short_name)


class ExploreOption2(ExploreOption):
    """ epsilon greedy selects Explore with probability p and one of the primitive
    actions at random with probability 1-p.
    This is to prevent the number of actions to have an impact on Explore selection.
    """

    def __init__(self, ex_prob, **kwargs):
        super(ExploreOption2, self).__init__(**kwargs)
        self.name = 'ExploreOption2'
        self.short_name = 'XOpt2'
        self.ex_prob = ex_prob # probability to select Explore in eps greedy

    def exploiter_act(self, obs):
        """ epsilon greedy with a twist: Explore is chosen with proba ex_prob"""
        if np.random.rand() < self.epsilon:
            if np.random.rand() < self.ex_prob:
                action = self.n_actions # explore option
            else:
                action = np.random.randint(self.n_actions) # a primitive action
        else:
            action = my_argmax(self.exploiter.Qtable[obs])

        return action

    def exploiter_act_no_opt(self, obs):
        """ Only to test the usefulness of the option choice.
        Removes the option from the choices of the exploiter in greedy """
        if np.random.rand() < self.epsilon:
            if np.random.rand() < self.ex_prob:
                action = self.n_actions # explore option
            else:
                action = np.random.randint(self.n_actions) # a primitive action
        else:
            action = my_argmax(self.exploiter.Qtable[obs][:-1]) # cannot choose option on purpose

        return action

    def act(self, obs):
        """ If the Explorer isn't on, the Meta acts. If it chooses the explore
        action, the Explorer takes over for the next c_switch steps """
        # obs = tuple(obs) # to access somewhere in the table
        if self.test_mode:
            action = self._act_no_option(obs)
            return action

        if self.exploring == 0: # meta acts
            if np.random.rand() <0.00001: print("\tMeta Q values: {}".format(self.exploiter.Qtable[obs]))
            action = self.exploiter_act(obs)
            #action = self.exploiter_act_no_opt(obs)
            self.explored = False

            if action == self.n_actions: # last action = explore
                self.obs_in_explo = obs # observation from which we explored
                self.exploring = self.c_switch

        if self.exploring > 0: # not 'else' since 'exploring' changes in if
            action = self.explorer.act(obs)
            self.explored = True # last action is an exploration
            self.exploring -= 1

        return action

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "lr={}, lrEO={}, gam={}, c_switch={}, p={}, agts:{}&{}"\
               .format(self.learn_rate, self.lrEO, self.gamma, self.c_switch, self.ex_prob, self.exploiter.short_name, self.explorer.short_name)


class ExploreAndExploit(Agent):
    """ Distribution picks the agents (whether explore or exploit) for the next
    c_switch steps. Both trained off-policy on all the data.
    """
    def __init__(self, env_shapes, exploiter_class, explorer_class, lbda=.9,
                 epsilon=0.1, learn_rate=2.5e-4, gamma=0.9, gamma_explo=0.9,
                 min_eps=0.1, min_eps_explo=0.25, c_switch=10, ex_prob=0.2,
                 **kwargs):
        super(ExploreAndExploit, self).__init__(env_shapes) # inits shapes
        self.name = 'ExploreAndExploit'
        self.short_name = 'XAX'
        print((*self.input_shape, self.n_actions))
        #self.explo_horizon = explo_horizon
        self.learn_rate = learn_rate
        self.lbda = lbda
        self.gamma = gamma
        self.c_switch = c_switch
        self.ex_prob = ex_prob # to choose one agent of the other for next c_switch steps
        self.exploiter = exploiter_class(env_shapes=env_shapes, lbda=lbda,
            learn_rate=self.learn_rate, gamma=self.gamma,
            explo_horizon=1, min_eps=min_eps)
        self.explorer = explorer_class(env_shapes=env_shapes, lbda=lbda,
            learn_rate=self.learn_rate, gamma=gamma_explo,
            explo_horizon=1, min_eps=min_eps_explo)
        self.epsilon = epsilon
        self.test_mode = False # i.e. we're in training mode by default
        self.reset()

    def epsilon():
        doc = "The epsilon property."
        def fget(self):
            return self._epsilon
        def fset(self, value):
            self._epsilon = value
            self.exploiter.epsilon = value
            self.explorer.epsilon = value
        def fdel(self):
            del self._epsilon
        return locals()
    epsilon = property(**epsilon())

    def reset(self):
        self.step = 0
        self.verbose = False
        self.rem_steps = 0 # remaining steps of acting for the current agent
        self.exploiter.reset()
        self.explorer.reset()
        self.visit_counts = np.zeros(self.input_shape)
        self.current_actor = self.exploiter

    def act(self, obs):
        """ If the Explorer isn't on, the Meta acts. If it chooses the explore
        action, the Explorer takes over for the next c_switch steps """
        # obs = tuple(obs) # to access somewhere in the table
        if self.test_mode:
            action = self.exploiter.act(obs)
            return action

        if self.rem_steps == 0:
            # Deciding the actor for the next steps
            if np.random.rand() < self.ex_prob:
                self.current_actor = self.explorer
            else:
                self.current_actor = self.exploiter
            self.rem_steps = self.c_switch

        action = self.current_actor.act(obs)
        self.rem_steps -= 1
        return action

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_transition.
        Updates the annealing epsilon. """
        if d:
            self.rem_steps = 0

        # Training both agents on- or off-policy
        #if not self.explored:
        self.exploiter.learn(s,a,r,s_,d)
        self.visit_counts[s_] += 1 # s is now visited
        explo_r = -0.01*np.sqrt(self.visit_counts[s_]) # the hyperparameter doesn't change anything
        self.explorer.learn(s, a, explo_r, s_) # no access to d

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "lr={}, ex_prob={}, c_switch={}, agts:{}&{}"\
               .format(self.learn_rate, self.ex_prob, self.c_switch, self.exploiter.short_name, self.explorer.short_name)


class QLearning_VC(QLearning):
    """ Improvement on QLearning with an exploration bonus with visit counts."""
    def __init__(self, beta=0.01, reward_function=Negative_sqrt, **kwargs):
        self.beta = beta
        self.reward_function = reward_function(**kwargs)
        super(QLearning_VC, self).__init__(**kwargs)
        self.name = 'QLearning_VC'
        self.short_name = 'QL+VC'

    def reset(self):
        super(QLearning_VC, self).reset()
        if not isinstance(self.reward_function, Negative_sqrt):
            self.Qtable += self.beta/(1-self.gamma) # optimistic
        self.reward_function.reset()
        self.visit_counts = np.ones(self.input_shape) # start counts at 1

    def learn(self, s, a, r, s_, d=False):
        """ Updates the Qtable based on the s,a,r,s_ transition.
        QLearning maxes over actions in the future state (off policy).
        Updates the annealing epsilon. """
        self.visit_counts[s_] += 1 # for display only
        #explo_r = 1/np.sqrt(self.visit_counts[s_])
        explo_r = self.reward_function.give_reward(s,a,s_)
        super(QLearning_VC, self).learn(s,a, r+self.beta*explo_r ,s_,d)

class Delayed_QLearning(Agent):
    """ Improvement on QLearning to perform averaging updates after waiting for
    a few steps. This algorithm is PAC-MDP."""
    def __init__(self, gamma=0.9, epsilon=0.1, delta=0.1, eps1=None, m=None, **kwargs):
        self.name = 'Delayed_QLearning'
        self.short_name = 'DelayQL'
        super(Delayed_QLearning, self).__init__(**kwargs) # inits shapes
        print((*self.input_shape, self.n_actions))
        sa = prod(self.input_shape)*self.n_actions
        self.gamma = gamma
        delta = 0.1
        if eps1 is None:
            self.eps1 = epsilon*(1-self.gamma)/9
        else:
            self.eps1 = eps1

        kappa = 1/((1-self.gamma)*self.eps1)
        if m is None:
            self.m = np.log(3*sa*(1+sa*kappa)/delta)/(2*self.eps1**2*(1-self.gamma)**2)
        else:
            self.m = m

        print("eps1 = {}; m = {}".format(self.eps1, self.m))
        self.reset() # variables init

    def reset(self):
        """ Algorithm initialization """
        self.Qtable = np.ones((*self.input_shape, self.n_actions))*1/(1-self.gamma)
        self.Utable = np.zeros_like(self.Qtable) # for attempted updates
        self.ltable = np.zeros_like(self.Qtable) # counters
        self.ttable = np.zeros_like(self.Qtable) # time of last attempted update
        self.LEARN  = np.ones_like(self.Qtable, dtype=bool) # LEARN flag
        self.t_star = 0 # time of the most recent Q-value change
        self.t = 0 # timestep


    def act(self, obs):
        """ Greedy policy over the Qtable """
        return my_argmax(self.Qtable[obs]) # break ties evenly

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
        self.t += 1
        r, discount = self._reward_seq_discounter(r) # outputs (r, gamma) unless option

        if self.LEARN[s][a]: # if you can LEARN
            # append to update memory U, maybe attempt update
            self.Utable[s][a] += r + discount*np.max(self.Qtable[s_])
            self.ltable[s][a] += 1
            if self.ltable[s][a] == self.m: # if you can attempt an update,
                # attempt an update
                if self.Qtable[s][a] - self.Utable[s][a]/self.m >= 2*self.eps1: # can you update?
                    # yes: do so!
                    self.Qtable[s][a] = self.Utable[s][a]/self.m + self.eps1
                    self.t_star = self.t # time of most recent Q value change
                elif self.ttable[s][a] >= self.t_star: # failed your update and already blew your timer
                    # you cannot LEARN until your timer is back later than t*
                    self.LEARN[s][a] = False

                self.ttable[s][a] = self.t # update timer (last attempted update)
                self.Utable[s][a] = 0 # restart accumulating updates for m steps...
                self.ltable[s][a] = 0 # counted by this guy

        elif self.ttable[s][a] < self.t_star: # if you can't LEARN but your timer is back later than t*
            # then you can LEARN again next time
            self.LEARN[s][a] = True


    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "gamma={}, eps1={}, m={}"\
               .format(self.gamma, self.eps1, self.m)
