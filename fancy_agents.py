import numpy as np
from utils import assert_not_abstract
from agents_core import Agent, QLearning

class HQL(Agent):
    """ Composed of 2 QL agents:
    1) Meta Controller, mapping states to goals (1to6 states),
        fed by extrinsic reward
    2) Controller, mapping (state,goal) pairs to atomic/primitive actions
        fed by the oracle's intrisic reward of 1_{state=goal}
     """

    def __init__(self, env_shapes, epsilon=0.1, learn_rate=2.5e-4, gamma=0.9):
        self.name = 'HQL'
        super(HQL, self).__init__(env_shapes)
        # Creating the controllers. We chose states as goals (handcrafted knowledge).
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.meta_controller = QLearning((
                self.input_shape, np.prod(self.input_shape)),
                learn_rate=learn_rate, explo_horizon=50000, gamma=gamma)
        self.controller = QLearning((
                (*self.input_shape, *self.input_shape), self.n_actions),
                learn_rate=learn_rate, gamma=gamma)
        self.steps_controller = 0
        self.steps_meta = -1  # reset increments it
        self.eps_meta = 1.
        self.eps_controller = np.ones(self.input_shape) # list of eps per goal
        self.success_counter = np.zeros(self.input_shape) # times task succeeded
        self.asked_counter = np.zeros(self.input_shape) # times task asked
        self.exp_counter = np.zeros(self.input_shape) # times task asked
        self._reset_meta()
        self.verbose = False

    def _reset_meta(self):
        """ Reset of the meta memory variables between each task """
        self.current_goal = None
        self.task_ongoing = False # no ongoing task
        self.cumulate_reward = 0  # meta-controller level reward

    def _decode(self, encoded_goal):
        """ Turns a 1D encoded goal into an input_shaped goal """
        # FOR NOW ONLY 2D GOALS (can't be arsed to find general formula .-.)
        row = encoded_goal // self.input_shape[1]
        col = encoded_goal - (self.input_shape[1]*row)
        return (row,col)

    def _encode(self, goal):
        """ Turns a goal into its 1D encoded version """
        # FOR NOW ONLY 2D GOALS (can't be arsed to find general formula .-.)
        return goal[0]*self.input_shape[1] + goal[1]

    def act(self, obs):
        """ If a task or goal is ongoing, just ask the controller to beat it,
        Otherwise the meta-controller picks a goal and the controller begins.
        """
        if not self.task_ongoing:
            self.meta_prev_state = obs
            encoded_goal = self.meta_controller.act(obs)
            self.current_goal = self._decode(encoded_goal)
            if self.verbose or np.random.rand()<0.001:
                print("Meta setted goal {}={}".format(encoded_goal,self.current_goal))
            self.task_ongoing = True

        action = self.controller.act(np.array([*obs, *self.current_goal]))
        return action

    def _update_eps_controller(self, success:bool, done:bool):
        """ Updates the eps of the current task according to:
        1) linearly decreasing from 1 to 0.1 over 50k steps normally
        2) set to 0.1 if goal success rate over 90% """
        self.exp_counter[self.current_goal] += 1
        self.asked_counter[self.current_goal] += int(done or success)
        self.success_counter[self.current_goal] += int(success)
        if (self.asked_counter[self.current_goal] > 0) and \
           (self.success_counter[self.current_goal]/self.asked_counter[self.current_goal] > 0.9):
            self.eps_controller[self.current_goal] = 0.1
        else:
            self.eps_controller[self.current_goal] = max((50000 - self.exp_counter[self.current_goal])/20000, 0.1)

    def _critic(self, state) -> float:
        """ Oracle: indicator of state-goal equality
        Function that returns the intrinsic reward. """
        return int(state==self.current_goal)

    def learn(self, s, a, r, s_, done=None):
        # Controller
        self.steps_controller += 1
        self.controller.epsilon = self.eps_controller[self.current_goal]
        intrinsic_r = self._critic(s_) # this is the oracle
        self.controller.learn((*s,*self.current_goal), a, intrinsic_r, (*s_,*self.current_goal))
        goal_reached = bool(intrinsic_r)
        self._update_eps_controller(goal_reached, done)
        # Meta controller
        self.cumulate_reward += r
        if goal_reached or done:
            # goal reached! Give reward, then reset variables
            if self.verbose and goal_reached: print("Goal reached! {}".format(self.current_goal))
            self.meta_controller.learn(self.meta_prev_state, self._encode(self.current_goal),
                                       self.cumulate_reward, s_)
            self._reset_meta()
            self.steps_meta += 1
            self.meta_controller.anneal_epsilon(self.steps_meta)

    def tell_specs(self):
        return "Annealing epsilon, learn_rate={}, gamma={}".format(self.learn_rate, self.gamma)

    def meta_policy(self):
        possible_states = [(i,j) for i in range(6) for j in range(2)]
        out = np.empty((6,2,2))
        for s in possible_states:
            out[s] = self._decode(np.argmax(self.meta_controller.Qtable[s]))
        return out

    def play_around(self, env, n_steps=100, max_reach=10):
        """ Lets the controller play around with the environment to figure out
        basic motions through Hinsight. """
        total_steps = 0
        sM = [] # states matrix
        aM = [] # actions matrix
        print("Playing around with the environment...")
        while total_steps < n_steps:
            obs = env.reset()
            sM.append(obs)
            for step in range(100):
                env.render()
                total_steps += 1
                action = env.action_space.sample()
                aM.append(action)
                obs, reward, done, info = env.step(action)
                sM.append(obs)
                if done:
                    env.render()
                    aM.append(None)
                    break
        env.close()
        print("Learning from this...")
#        print("State and Action matrices of sizes {} and {} looking like \n{} \nand \n{}".format(len(sM), len(aM), sM, aM))
        for i in range(len(sM)-1, 0, -1): # starting from the end
            for l in range(1,max_reach):
                if aM[i-l] is None:
                    break # the episode changed
                current_s = sM[i-l]
                target_s  = sM[i]
                self.controller.learn(
                    (*current_s,*target_s), # s True only with gamma=1 and reward at the end.
                    aM[i-l],                # a
                    1,                      # r
                    (*target_s,*target_s))  # s'
