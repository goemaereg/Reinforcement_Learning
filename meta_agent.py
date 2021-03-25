from tabular.agents import QLearning
from gym import spaces
from utils import assert_not_abstract, my_argmax
from model import Model
import numpy as np
import random
from gym.envs.registration import register
import matplotlib.pyplot as plt
import matplotlib.colors as clr


register(
    id='FourRoomsKeyDoorEnv-v0',
    entry_point='gym_additions.envs:FourRoomsKeyDoorEnv',
    )
register(
    id='FourRoomsBigKeyDoorEnv-v0',
    entry_point='gym_additions.envs:FourRoomsBigKeyDoorEnv',
    )

env_big = False
# env_big = True

if env_big:
    env_name = 'FourRoomsBigKeyDoorEnv-v0'
    ctrl_agent_path = 'outputm/train_stl_28_HER_FourRoomsGoalBig-v0_QLearning_perf_10.train.agent.npy'
    ctrl_model_name = 'train_stl_44'
    meta_agent_path = 'outputm/meta_Meta_FourRoomsBigKeyDoorEnv-v0_MetaAgent_perf_10.agent.npy'
else:
    env_name = 'FourRoomsKeyDoorEnv-v0'
    ctrl_agent_path = 'outputm/train_stl_28_HER_FourRoomsGoal-v0_QLearning_perf_3.train.agent.npy'
    ctrl_model_name = 'train_stl_28'
    meta_agent_path = 'outputm/meta_Meta_FourRoomsKeyDoorEnv-v0_MetaAgent_perf_3.agent.npy'

args_ctrl_agent = dict(model_name=ctrl_model_name,
                agent_class=QLearning, env_name=env_name,
                env_big=env_big, subtraject_len=44)


class MetaAgent(QLearning):
    """ Improvement on Sarsa to max Q(S',.) over all actions """
    def __init__(self, actions, **kwargs):
        super(MetaAgent, self).__init__(**kwargs)
        self.name = 'MetaAgent'
        self.short_name = 'MQL'
        self.actions = actions

    # def reset(self):
    #     self.Qtable = np.zeros((self.input_shape, self.n_actions)) # arb init at 0
    #     self.verbose = False
    #     self.reset_eps()

    def act(self, obs):
        """ Epsilon-greedy policy over the Qtable """
        # obs = tuple(obs) # to access somewhere in the table
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, len(self.actions))
            random_position = self.actions[action]
            return random_position
        else:
            if self.verbose:
                print("Q values of possible actions: {}".format(self.Qtable[obs]))
            action = my_argmax(self.Qtable[obs])
            return self.actions[action]

    def learn(self, s, a, r, s_, d=False):
        # unravel action (position)
        a0 = self.actions.index(a)
        super(MetaAgent, self).learn(s, a0, r, s_, d=False)


class MetaModel(Model):
    def __init__(self, model_ctrl, **kwargs):
        super(MetaModel, self).__init__(name='Meta', **kwargs)
        self.model_ctrl = model_ctrl

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
        opt_history = np.zeros(episodes)
        key_history = np.zeros(episodes)
        key = lambda obs: obs[2]
        for ep in range(episodes):
            if ep > 0:
                opt_history[ep] = opt_history[ep - 1]

            obs = self.model_ctrl.env.reset()
            # action = self.agent.act(obs)
            steps = 0
            goals = 0
            reward = 0
            done = False
            # make sure control agent acts greedy:
            old_eps = self.model_ctrl.agent.epsilon
            self.model_ctrl.agent.epsilon = 0

            for _ in range(max_episode_steps):
                meta_goal = self.agent.act(key(obs))
                old_obs = obs  # tuples don't have the copy problem
                ctrl_steps = 0
                for _ in range(500):

                    ctrl_agent_obs = (*self.model_ctrl.env.s, *meta_goal)
                    ctrl_action = self.model_ctrl.agent.act(ctrl_agent_obs)
                    # act in env, i.e. use action as goal in controller agent (model)
                    obs, reward, done, _ = self.model_ctrl.env.step(ctrl_action)
                    ctrl_steps += 1
                    # goal reached ?
                    if self.model_ctrl.env.s != meta_goal:
                        # key picked up by chance?
                        if not key(old_obs) and key(obs):
                            key_history[ep] = 1
                    else:
                        break

                steps += ctrl_steps
                self.agent.learn(key(old_obs), meta_goal, reward, key(obs), done)
                goals += 1
                if done:
                    break
            opt_history[ep] = ep #+= goals
            steps_history[ep] = goals
            if (ep % 500) == 0 or ep == (episodes - 1):
                print(
                    f'ep: {ep} goals: {goals} '
                    f'actions: {steps} reward: {reward} '
                    f'key: {key(obs)} door: {int(done)}')
            # undo control agent greedy mode
            self.model_ctrl.agent.epsilon = old_eps

        self.xaxis = opt_history
        self.yaxis = steps_history
        return opt_history, steps_history, key_history

    def train_runs(self, runs=10, episodes=3000, max_episode_steps=10000):
        """
        Run the agent in environment

        Args:
            runs (int): Maximum number of runs
            episodes (int): Number of episodes to run
            max_episode_steps (int): Maximum number steps to run in each episode
        """
        opt_history_lst = []
        steps_history_lst = []
        key_history_lst = []
        np.random.seed(0)
        random.seed(0)
        for run in range(runs):
            # Create new agent instance for every run to drop learned experience.
            self.agent = self.agent_class(**self.agent_args)
            opt, steps, keys = self.train(episodes=episodes,
                                     max_episode_steps=max_episode_steps)
            opt_history_lst.append(opt)
            steps_history_lst.append(steps)
            key_history_lst.append(keys)
            # print("Final performance: {}".format(perf[-1]))
        opt_history = np.mean(opt_history_lst, axis=0)
        steps_history = np.mean(steps_history_lst, axis=0)
        key_history = np.mean(key_history_lst, axis=0)
        self.xaxis = opt_history
        self.yaxis = steps_history
        return opt_history, steps_history, key_history

    def test(self, episodes, max_episode_steps):
        # make sure meta agent acts greedy:
        meta_old_eps = self.agent.epsilon
        self.agent.epsilon = 0
        # make sure control agent acts greedy:
        ctrl_old_eps = self.model_ctrl.agent.epsilon
        self.model_ctrl.agent.epsilon = 0

        # Steps history per episode
        steps_history = np.zeros(episodes)
        # Optimizatons history per episode
        opt_history = np.zeros(episodes)
        # Key picked up by chance
        key_history = np.zeros(episodes)
        key = lambda obs: obs[2]
        pos = lambda obs: obs[:2]
        policies = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

        for ep in range(episodes):
            obs = self.model_ctrl.env.reset()
            goals = 0
            done = False
            actions = 0
            for _ in range(max_episode_steps):
                meta_action = self.agent.act(key(obs))
                done = False
                for _ in range(500):
                    ctrl_agent_obs = (*self.model_ctrl.env.s, *meta_action)
                    # act in env, i.e. use action as goal in controller agent (model)
                    ctrl_action = self.model_ctrl.agent.act(ctrl_agent_obs)
                    old_obs = obs
                    obs, _, done, _ = self.model_ctrl.env.step(ctrl_action)
                    # print(f'{pos(old_obs)} {policies[ctrl_action]} -> {pos(obs)}')
                    actions += 1
                    # goal reached ?
                    if self.model_ctrl.env.s != meta_action:
                        # key picked up by chance?
                        if not key(old_obs) and key(obs):
                            key_history[ep] = 1
                    else:
                        break

                goals += 1
                print(f'ep: {ep} goals: {goals} actions: {actions} key:{key(obs)} door:{int(done)}')
                if done:
                    break
            opt_history[ep] = ep
            steps_history[ep] = goals

        # undo control agent greedy mode
        self.agent.epsilon = meta_old_eps
        # undo control agent greedy mode
        self.model_ctrl.agent.epsilon = ctrl_old_eps
        self.xaxis = opt_history
        self.yaxis = steps_history
        return opt_history, steps_history, key_history

    def save_policy_plot(self, path=None, text=True):
        """ Visualizes a policy and value function given agent and environment."""
        cmap = clr.LinearSegmentedColormap.from_list('mycmap',
                                                         ['#FF0000',
                                                          '#000000',
                                                          '#008000'])
        for key in range(self.agent.input_shape[0]):
            grid = np.zeros((self.env.height, self.env.width))
            high = np.max(self.agent.Qtable[key])
            for obs in self.env.obstacles:
                grid[obs[0]][obs[1]] = -high
            fig, ax = plt.subplots()
            for i in range(len(self.agent.actions)):
                height, width = self.agent.actions[i]
                grid[height][width] = self.agent.Qtable[key][i]
                if text:
                    ax.text(width, height,
                               f'{self.agent.Qtable[key][i]:4.2f}',
                               ha='center', va='center', color='w',
                            fontsize='xx-small')
            im = ax.imshow(grid, cmap=cmap)
            ax.set_title(f'Meta QValue visualization (key = {key})')
            fig.tight_layout()
            path = f'{self.path}.qtable.{key}.plot.png'
            plt.savefig(path)


def create_meta_model(model_ctrl):
    # state space: has_key = int(boolean)
    observation_space = spaces.Discrete(2)
    # action space: position = tuple(x = range(0:width), y = range(0:height)
    # unravel position space into one-dimensional action space
    actions = []
    for height in range(model_ctrl.env.height):
        for width in range(model_ctrl.env.width):
            if (height, width) not in model_ctrl.env.obstacles:
                actions.append((height, width))
    action_space = spaces.Discrete(len(actions))
    shapes = ((observation_space.n,), action_space.n)

    agent_args = {
        'env_shapes': shapes,
        'min_eps': 0.1,
        'explo_horizon': 1,
        'learn_rate': 0.15,
        'explo_steps': 10,
        'gamma': 0.9,
        'lambda': 0.9,
        'n': 10,
        'actions': actions
    }
    args = dict(model_name=f'meta',
                model_ctrl=model_ctrl,
                agent_class=MetaAgent,
                agent_args=agent_args,
                env_name=env_name,
                env=model_ctrl.env,
                env_big=env_big)
    model_meta = MetaModel(**args)
    return model_meta


def train_meta_model(model, episodes):
    ep, goals, keys = model.train_runs(episodes=episodes, max_episode_steps=10000)
    # ep, goals, keys = model.train(episodes=episodes, max_episode_steps=10000)
    model.save_agent(f'{model.path}.agent.npy')
    model.save_plot_data(f'{model.path}.train.plot.npy')
    model.save_plot(f'{model.path}.train.plot', episodes=episodes,
                         yscale=None, ybase=2, smooth=True,
                         xlabel='Episodes', ylabel='Goals',
                         xaxis=ep, yaxis=goals, xlineat=2)
    model.save_plot(f'{model.path}.train.key.plot', smooth=False,
                    title='Key pick-up by chance',
                    xlabel='Episode', ylabel='Key picked-up',
                    xaxis=ep, yaxis=keys)


def main():
    model_ctrl = Model(**args_ctrl_agent)
    model_ctrl.load_agent(ctrl_agent_path)
    #print(model.test())
    # meta agent
    train_episodes=3000
    model_meta = create_meta_model(model_ctrl=model_ctrl)
    train_meta_model(model_meta, episodes=train_episodes)
    # model_meta.load_agent(meta_agent_path)
    # test_episodes = 100
    # model_meta.load_agent(f'{model_meta.path}.agent.npy')
    # ep, goals, keys = model_meta.test(episodes=test_episodes, max_episode_steps=100)
    # model_meta.save_plot(f'{model_meta.path}.test.plot', episodes=test_episodes,
    #                 yscale=None, ybase=2, smooth=False,
    #                 xlabel='Episodes', ylabel='Goals')
    # model_meta.save_plot(f'{model_meta.path}.test.key.plot',
    #                      smooth=False, title='Key pick-up by chance',
    #                      xlabel='Episode', ylabel='Key picked-up',
    #                      xaxis=ep, yaxis=keys)
    # model_meta.save_policy_plot()

if __name__ == '__main__':
    main()