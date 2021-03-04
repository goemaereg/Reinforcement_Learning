from tabular.agents import QLearning
from gym import spaces
from utils import assert_not_abstract, my_argmax
from model import Model
import numpy as np
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
    path_ctrl_agent = 'output/her_tabular_FourRoomsGoalBig-v0_QLearning_perf_n_subtraject_steps_44_10.npy'
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

        for ep in range(episodes):
            if ep > 0:
                opt_history[ep] = opt_history[ep - 1]

            self.env.reset()
            obs = int(self.env.has_key)
            # action = self.agent.act(obs)
            steps = 0
            reward, done, info = 0, False, {}
            for _ in range(max_episode_steps):
                action = self.agent.act(obs)
                old_obs = obs  # tuples don't have the copy problem
                # act in env, i.e. use action as goal in controller agent (model)
                ctrl_steps, reward, done, info = self.model_ctrl.task(goal=action,
                                                      max_episode_steps=100)
                steps += ctrl_steps
                obs = int(info['key'])
                self.agent.learn(old_obs, action, reward, obs, done)
                if done:
                    break
            opt_history[ep] = ep
            steps_history[ep] = steps
            key = int(info.get('key', False))
            if ep < 10 or (ep % 500) == 0 or ep == (episodes - 1):
                print(
                    f'ep: {ep} steps: {steps} reward: {reward} key: {key} door: {int(done)}')

        self.xaxis = opt_history
        self.yaxis = steps_history
        return self.xaxis, self.yaxis


def create_meta_model(model_ctrl):
    # state space: has_key = int(boolean)
    observation_space = spaces.Discrete(2)
    # action space: position = tuple(x = range(0:width), y = range(0:height)
    # unravel position space into one-dimensional action space
    actions = []
    for x in range(model_ctrl.env.width):
        for y in range(model_ctrl.env.height):
            if [x, y] not in model_ctrl.env.obstacles:
                actions.append((x, y))
    action_space = spaces.Discrete(len(actions))
    shapes = ((observation_space.n,), action_space.n)

    agent_args = {
        'env_shapes': shapes,
        'explo_horizon': 1,
        'learn_rate': 0.05,
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


def main():
    model_ctrl = Model(**args_ctrl_agent)
    model_ctrl.load_agent(path_ctrl_agent)
    #print(model.test())
    # meta agent
    model_meta = create_meta_model(model_ctrl=model_ctrl)
    model_meta.train(episodes=5000, max_episode_steps=100)
    model_meta.save_plot(f'{model_meta.path}.plot')


if __name__ == '__main__':
    main()