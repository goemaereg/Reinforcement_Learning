from tabular.agents import QLearning
from sklearn.model_selection import ParameterGrid
from model import HERModel


# env_big = True
env_big = False
env_name = 'FourRoomsGoalBig-v0' if env_big else 'FourRoomsGoal-v0'

# grid = ParameterGrid({'subtraject_len': range(4, 64, 8)})
grid = ParameterGrid({'subtraject_len': range(28, 29, 8)})


if __name__ == '__main__':
    runs=1
    for params in grid:
        args = dict(model_name=f'train_stl_{params["subtraject_len"]}',
                    agent_class=QLearning, env_name=env_name,
                    env_big=env_big, **params)
        model = HERModel(**args)
        print(model.env.render())
        print(model.env.obstacles)
        episodes = 20000 if env_big else 3000
        model.train_runs(runs=runs, episodes=episodes)
        model.save_agent(f'{model.path}.train.agent.npy')
        model.save_plot_data(f'{model.path}.train.plot.npy')
        model.save_plot(f'{model.path}.train.plot')

        test_episodes = 10
        model.test(episodes=test_episodes, max_episode_steps=100)
        model.save_plot(f'{model.path}.test.plot',
                        episodes=test_episodes,
                        yscale=None, smooth=False,
                        xlabel='Episodes', ylabel='Actions')
