import gym
import gym_additions
# from tabular.agents import *
from collections import deque
from utils import save_plot
import numpy as np
import random


class Model():
    """ Reinforcement Learning Model

    A Model is an abstract entity that can simply act in an environment.
    """
    def __init__(self, model_name, agent_class, env_name, name='Base', **kwargs):
        """
        Initializes the model

        Args:
            agent_class (class) : The class of the agent
            env_name (str): OpenAI Gym environment name
            **kwargs (dict): Model keyword arguments
        """
        self.name = name
        self.model_name = model_name
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        shapes = (tuple([s.n for s in self.env.observation_space]),
                  self.env.action_space.n)

        self.agent_class = agent_class
        self.agent_args = {
            'env_shapes': shapes,
            'explo_horizon': 1,
            'learn_rate': 0.05,
            'explo_steps': 10,
            'gamma': 0.9,
            'lambda': 0.9,
            'n': 10
        }
        self.agent = self.agent_class(**self.agent_args)
        self.xaxis = None
        self.yaxis = None

        launchspecs = f'perf_{self.env.roomsize}'
        label = f'{self.model_name}_{self.name}_{self.env_name}_{self.agent.name}_{launchspecs}'
        self.path = f'outputm/{label}'

    def train(self, episodes, max_episode_steps):
        """
        Train the agent in environment

        Args:
            episodes (int): Number of episodes to run
            max_episode_steps (int): Maximum number steps to run in each episode
        """
        # Steps history per episode
        steps_history = np.zeros(episodes)
        # Optimizatons history per episode
        opt_history = np.empty(episodes)
        for ep in range(episodes):
            if ep > 0:
                opt_history[ep] = opt_history[ep - 1]
            else:
                opt_history[ep] = 0
            obs = self.env.reset()
            step = 0
            for step in range(max_episode_steps):
                action = self.agent.act(obs)
                old_obs = obs  # tuples don't have the copy problem
                obs, reward, done, info = self.env.step(action)
                self.agent.learn(old_obs, action, reward, obs, done)
                opt_history[ep] += 1
                if done:
                    break

                steps_history[ep] = step
            if ep == 0:
                print(f'First trial in {step} steps')
            elif ep == (episodes - 1):
                print(f'Final trial in {step} steps')
            elif ep % (episodes // 5) == 0:
                print(f'Trial {ep:>5} in {step} steps')
        self.env.close()
        self.xaxis = opt_history[1:]
        self.yaxis = steps_history[1:]  # first is purely random
        return self.xaxis, self.yaxis

    def test(self, episodes=10, max_episode_steps=10000):
        """
        Test the agent in environment

        Args:
            episodes (int): Number of episodes to run
            max_episode_steps (int): Maximum number steps to run in each episode
        """
        # Testing phase
        # self.agent.verbose = True
        old_eps = self.agent.epsilon
        self.agent.epsilon = 0
        # Steps history per episode
        steps_history = np.zeros(episodes)
        rewards_history = np.zeros(episodes)
        for ep in range(episodes):
            cumreward = 0
            obs = self.env.reset()
            step = 0
            for step in range(max_episode_steps):
                action = self.agent.act(obs)
                old_obs = obs  # tuples don't have the copy problem
                obs, reward, done, info = self.env.step(action)
                cumreward += reward
                if done:
                    break

                steps_history[ep] = step
            rewards_history[ep] = cumreward
        self.env.close()
        self.agent.verbose = False
        self.agent.epsilon = old_eps
        self.xaxis = range(episodes)
        self.yaxis = steps_history
        return self.xaxis, self.yaxis

    def train_runs(self, runs=10, episodes=3000, max_episode_steps=150000):
        """
        Run the agent in environment

        Args:
            runs (int): Maximum number of runs
            episodes (int): Number of episodes to run
            max_episode_steps (int): Maximum number steps to run in each episode
        """
        perf_lst = []
        opt_lst = []
        np.random.seed(0)
        random.seed(0)
        for run in range(runs):
            # Create new agent instance for every run to drop learned experience.
            self.agent = self.agent.__class__(**self.agent_args)
            xaxis, perf = self.train(episodes=episodes,
                                     max_episode_steps=max_episode_steps)
            perf_lst.append(perf)
            opt_lst.append(xaxis)
            # print("Final performance: {}".format(perf[-1]))
        opt = np.mean(opt_lst, axis=0)
        perf = np.mean(perf_lst, axis=0)
        self.xaxis = opt
        self.yaxis = perf
        return opt, perf

    def load_agent(self, filename):
        self.agent.Qtable = np.load(filename)

    def save_agent(self, path):
        np.save(path, self.agent.Qtable)

    def load_plot_data(self, path):
        self.agent.Qtable = np.load(path)

    def save_plot_data(self, path):
        arr = np.array([self.xaxis, self.yaxis])
        np.save(path, arr)

    def save_plot(self, path, episodes=3000):
        # calculate smoothed plot (just to determine axis scale)
        smooth_avg = episodes // 100
        l_smooth = [None for _ in range(smooth_avg)]
        l_smooth += [np.mean(self.yaxis[i - smooth_avg:i + smooth_avg])
                     for i in range(smooth_avg, self.yaxis.size - smooth_avg)]
        # plot scales
        n_plot_xscale = (0, self.xaxis.max())
        n_plot_yscale = (0, max(l_smooth[smooth_avg:]))
        suptitle = f'{self.agent.name} performance on {self.env_name[:-3]}{self.env.roomsize}'
        title = self.agent.tell_specs()
        xlabel = 'Optimisation steps'
        ylabel = "Performance at {}".format(self.env_name)
        save_plot(self.yaxis, path, suptitle, title, xlabel, ylabel,
                  xaxis=self.xaxis,
                  interval_xaxis=n_plot_xscale,
                  interval_yaxis=n_plot_yscale,
                  smooth_avg=episodes//100, only_avg=True)


class ReplayBuffer(object):
    """  Replay experience buffer """
    def __init__(self, capacity):
        """
        Args:
            capacity (int): The length of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Args:
            batch_size (int): The number of samples
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class HERModel(Model):
    def __init__(self, env_big=False, subtraject_len=0, **kwargs):
        super(HERModel, self).__init__(name='HER', **kwargs)

        self.env_big = env_big
        self.buffer_size = 1 * 1024 if not self.env_big else 4 * 1024
        # number of optimization cycles within an episode
        self.mini_batches = 2 if not self.env_big else 16
        # sample batch size for each optimization cycle
        self.batch_size = 8 if not self.env_big else 64
        self.subtraject_len = 28 if not self.env_big else 44
        if subtraject_len > 0:
            self.subtraject_len = subtraject_len

    @staticmethod
    def env_state(obj):
        return (obj[0], obj[1])

    @staticmethod
    def env_goal(obj):
        return (obj[2], obj[3])

    def train(self, episodes, max_episode_steps):
        """
        Train the agent in environment

        Args:
            episodes (int): Number of episodes to run
            max_episode_steps (int): Maximum number steps to run in each episode
        """
        replay_buffer = ReplayBuffer(self.buffer_size)
        # Steps history per episode
        steps_history = np.zeros(episodes)
        # Optimizatons history per episode
        opt_history = np.empty(episodes)
        for ep in range(episodes):
            # if (ep%(n_episodes//5)==0):
            #     print("Episode {}/{}".format(ep+1, n_episodes))
            # sample initial state and goal
            obs = self.env.reset()
            transition_history = []
            # log an episode
            for step in range(max_episode_steps):
                # sample action a using behavioral policy from agent
                action = self.agent.act(obs)
                # execute action and observe new state
                new_obs, reward, done, _ = self.env.step(action)
                # use a binary and sparse reward from the environment: int(done)
                # store transition in replay buffer
                transition = (obs, action, reward, new_obs, done)
                replay_buffer.push(*transition)
                transition_history.append(transition)
                # update obs
                obs = new_obs
                steps_history[ep] = step
                if done:
                    break

            step = 0
            for step in range(len(transition_history)):
                # sample a set of additional goals for replay in this sub trajectory: # S(so, s1, ..., sT) = m(sT) = sT
                goal_index = (((step // self.subtraject_len) + 1) * self.subtraject_len) - 1
                goal_index = min(goal_index, len(transition_history) - 1)
                _, _, _, goal_obs, _ = transition_history[goal_index]
                sampled_goal = HERModel.env_state(goal_obs)
                # current transition step
                obs, action, _, new_obs, _ = transition_history[step]
                state = HERModel.env_state(obs)
                new_state = HERModel.env_state(new_obs)
                # use a binary and sparse reward if new goal reached
                done = (state == sampled_goal)
                # reward = - int(done == False)
                reward = int(done)
                # store transition in replay buffer
                transition = ((*state, *sampled_goal), action, reward,
                              (*new_state, *sampled_goal), done)
                replay_buffer.push(*transition)

            # learn on N cycles of minibatches of size B
            if ep > 0:
                opt_history[ep] = opt_history[ep - 1]
            else:
                opt_history[ep] = 0
            for _ in range(self.mini_batches):
                # Sample a minibatch B from the replay buffer
                n_samples = min(self.batch_size, len(replay_buffer))
                transitions = replay_buffer.sample(n_samples)
                # Perform one step of learning using the agent and the minibatch
                for transition in transitions:
                    self.agent.learn(*transition)

                opt_history[ep] += n_samples

            step = steps_history[ep]
            if ep == 0:
                print(f'First trial in {step} steps')
            elif ep == (episodes - 1):
                print(f'Final trial in {step} steps')
            elif ep % (episodes // 5) == 0:
                print(f'Trial {ep + 1:>5} in {step} steps')

        self.env.close()
        self.xaxis = opt_history[1:]
        self.yaxis = steps_history[1:]  # first is purely random
        return self.xaxis, self.yaxis
