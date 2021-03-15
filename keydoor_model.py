from tabular.agents import QLearning
from gym import spaces
from gym.envs.registration import register
from model import Model
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from utils import my_argmax


env_big = False
# env_big = True

register(
    id='FourRoomsKeyDoorEnv-v0',
    entry_point='gym_additions.envs:FourRoomsKeyDoorEnv',
    )
register(
    id='FourRoomsBigKeyDoorEnv-v0',
    entry_point='gym_additions.envs:FourRoomsBigKeyDoorEnv',
    )

if env_big:
    env_name = 'FourRoomsBigKeyDoorEnv-v0'
else:
    env_name = 'FourRoomsKeyDoorEnv-v0'

args_model = dict(model_name='keydoor',
                agent_class=QLearning, env_name=env_name,
                env_big=env_big)


class KeyDoorModel(Model):
    def __init__(self, **kwargs):
        super(KeyDoorModel, self).__init__(name='KeyDoor', **kwargs)

    def save_policy_plot(self, path=None, text=True):
        """ Visualizes a policy and value function given agent and environment."""
        cmap = clr.LinearSegmentedColormap.from_list('mycmap',
                                                         ['#FF0000',
                                                          '#000000',
                                                          '#008000'])
        policies = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        for key in range(self.agent.input_shape[-1]):
            grid = np.zeros((self.env.height, self.env.width))
            # high = np.max(self.agent.Qtable[key])
            # for obs in self.env.obstacles:
            #     grid[obs[0]][obs[1]] = -high
            fig, ax = plt.subplots()
            high = 0
            for height in range(self.env.height):
                for width in range(self.env.width):
                    if [height, width] in self.env.obstacles:
                        continue
                    obs = (height, width, key)
                    action = my_argmax(self.agent.Qtable[obs])
                    value = self.agent.Qtable[obs][action]
                    if value > high:
                        high = value
                    policy = policies[action]
                    grid[height][width] = value
                    if text:
                        fontsize = 'xx-small' if env_big else 'medium'
                        ax.text(width, height,
                                f'{policy}\n{value:4.2f}',
                                ha='center', va='center', color='w',
                                fontsize=fontsize)
            for obs in self.env.obstacles:
                grid[obs[1]][obs[0]] = -high

            im = ax.imshow(grid, cmap=cmap)
            ax.set_title(f'QValue visualization (key = {key})')
            fig.tight_layout()
            path = f'{self.path}.qtable.{key}.plot.png'
            plt.savefig(path)


def main():
    model = KeyDoorModel(**args_model)
    train_episodes = 3000
    optimal = 60 if env_big else 18
    model.train_runs(episodes=train_episodes, max_episode_steps=10000)
    model.save_agent(f'{model.path}.train.agent.npy')
    model.save_plot(f'{model.path}.train.plot', episodes=train_episodes,
                    xaxis=list(range(train_episodes))[1:],
                    xlabel='Episodes', ylabel='Actions',
                    xlineat=optimal)
    test_episodes = 100
    model.load_agent(f'{model.path}.train.agent.npy')
    model.test(episodes=test_episodes, max_episode_steps=100)
    model.save_plot(f'{model.path}.test.plot', episodes=test_episodes,
                    smooth=False)
    model.save_policy_plot()


if __name__ == '__main__':
    main()