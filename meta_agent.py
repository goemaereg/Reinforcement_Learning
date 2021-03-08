from tabular.agents import QLearning
from gym import spaces
from utils import assert_not_abstract, my_argmax
from model import Model
import numpy as np
import random
from gym.envs.registration import register

register(
    id='FourRoomsKeyDoorEnv-v0',
    entry_point='gym_additions.envs:FourRoomsKeyDoorEnv',
    )
register(
    id='FourRoomsBigKeyDoorEnv-v0',
    entry_point='gym_additions.envs:FourRoomsBigKeyDoorEnv',
    )

# env_big = False
env_big = True

if env_big:
    env_name = 'FourRoomsBigKeyDoorEnv-v0'
    path_ctrl_agent = 'outputm/train_stl_44_HER_FourRoomsGoalBig-v0_QLearning_perf_10.agent.npy'
    model_name = 'train_stl_44',
else:
    env_name = 'FourRoomsKeyDoorEnv-v0'
    path_ctrl_agent = 'outputm/train_stl_28_HER_FourRoomsGoal-v0_QLearning_perf_3.agent.npy'
    model_name = 'train_stl_28',

args_ctrl_agent = dict(model_name=model_name,
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
        key = lambda obs: obs[2]
        for ep in range(episodes):
            if ep > 0:
                opt_history[ep] = opt_history[ep - 1]

            obs = self.model_ctrl.env.reset()
            # action = self.agent.act(obs)
            steps = 0
            tasks = 0
            reward = 0
            done = False
            # make sure control agent acts greedy:
            old_eps = self.model_ctrl.agent.epsilon
            self.model_ctrl.agent.epsilon = 0

            for _ in range(max_episode_steps):
                meta_action = self.agent.act(key(obs))
                old_obs = obs  # tuples don't have the copy problem
                ctrl_steps = 0
                for _ in range(500):

                    ctrl_agent_obs = (*self.model_ctrl.env.s, *meta_action)
                    ctrl_action = self.model_ctrl.agent.act(ctrl_agent_obs)
                    # act in env, i.e. use action as goal in controller agent (model)
                    obs, reward, done, _ = self.model_ctrl.env.step(ctrl_action)
                    ctrl_steps += 1
                    # goal reached ?
                    if meta_action == self.model_ctrl.env.s:
                        break

                steps += ctrl_steps
                self.agent.learn(key(old_obs), meta_action, reward, key(obs), done)
                tasks += 1
                if done:
                    break
            opt_history[ep] = ep #+= tasks
            steps_history[ep] = tasks
            if (ep % 500) == 0 or ep == (episodes - 1):
                print(
                    f'ep: {ep} tasks: {tasks} '
                    f'actions: {steps} reward: {reward} '
                    f'key: {key(obs)} door: {int(done)}')
            # undo control agent greedy mode
            self.model_ctrl.agent.epsilon = old_eps

        self.xaxis = opt_history
        self.yaxis = steps_history
        return self.xaxis, self.yaxis

    def train_runs(self, runs=10, episodes=3000, max_episode_steps=10000):
        """
        Run the agent in environment

        Args:
            runs (int): Maximum number of runs
            episodes (int): Number of episodes to run
            max_episode_steps (int): Maximum number steps to run in each episode
        """
        xaxis_lst = []
        yaxis_lst = []
        np.random.seed(0)
        random.seed(0)
        for run in range(runs):
            # Create new agent instance for every run to drop learned experience.
            self.agent = self.agent_class(**self.agent_args)
            xaxis, yaxis = self.train(episodes=episodes,
                                     max_episode_steps=max_episode_steps)
            xaxis_lst.append(xaxis)
            yaxis_lst.append(yaxis)
            # print("Final performance: {}".format(perf[-1]))
        self.xaxis = np.mean(xaxis_lst, axis=0)
        self.yaxis = np.mean(yaxis_lst, axis=0)
        return self.xaxis, self.yaxis

    def test(self, max_episode_steps):
        # make sure meta agent acts greedy:
        meta_old_eps = self.agent.epsilon
        self.agent.epsilon = 0
        # make sure control agent acts greedy:
        ctrl_old_eps = self.model_ctrl.agent.epsilon
        self.model_ctrl.agent.epsilon = 0

        obs = self.model_ctrl.env.reset()
        tasks = 0
        done = False
        key = lambda obs: obs[2]
        step = 0
        for _ in range(max_episode_steps):
            meta_action = self.agent.act(key(obs))
            done = False
            for step in range(500):
                ctrl_agent_obs = (*self.model_ctrl.env.s, *meta_action)
                # act in env, i.e. use action as goal in controller agent (model)
                ctrl_action = self.model_ctrl.agent.act(ctrl_agent_obs)
                obs, _, done, _ = self.model_ctrl.env.step(ctrl_action)
                # goal reached ?
                if meta_action == self.model_ctrl.env.s:
                    break

            tasks += 1
            print(f'task: {tasks} steps: {step} key:{key(obs)} door:{int(done)}')
            if done:
                break

        # undo control agent greedy mode
        self.agent.epsilon = meta_old_eps
        # undo control agent greedy mode
        self.model_ctrl.agent.epsilon = ctrl_old_eps
        return tasks, obs[2], done


def create_meta_model(model_ctrl):
    # state space: has_key = int(boolean)
    observation_space = spaces.Discrete(2)
    # action space: position = tuple(x = range(0:width), y = range(0:height)
    # unravel position space into one-dimensional action space
    actions = []
    for height in range(model_ctrl.env.height):
        for width in range(model_ctrl.env.width):
            if [height, width] not in model_ctrl.env.obstacles:
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
    model.train_runs(episodes=episodes, max_episode_steps=10000)
    model.save_agent(f'{model.path}.agent.npy')
    model.save_plot_data(f'{model.path}.plot.npy')
    model.save_plot(f'{model.path}.plot', episodes=episodes,
                         yscale='log', ybase=2, smooth=True,
                         xlabel='Episodes', ylabel='Tasks')

def test_meta_model(model, max_episode_steps=10000):
    for _ in range(2):
        np.random.seed(0)
        random.seed(0)
        tasks, key, done = model.test(max_episode_steps=max_episode_steps)
        print (f'tasks: {tasks} key: {key}: door: {int(done)}')


def main():
    model_ctrl = Model(**args_ctrl_agent)
    model_ctrl.load_agent(path_ctrl_agent)
    #print(model.test())
    # meta agent
    episodes=3000
    model_meta = create_meta_model(model_ctrl=model_ctrl)
    train_meta_model(model_meta, episodes)
    test_meta_model(model_meta, max_episode_steps=10000)

if __name__ == '__main__':
    main()