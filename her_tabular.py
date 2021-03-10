from tabular.agents import QLearning
from sklearn.model_selection import ParameterGrid
from model import HERModel


env_big = True
# env_big = False
env_name = 'FourRoomsGoalBig-v0' if env_big else 'FourRoomsGoal-v0'

# grid = ParameterGrid({'subtraject_len': range(4, 64, 8)})
grid = ParameterGrid({'subtraject_len': range(44,48, 8)})


if __name__ == '__main__':
    runs=1
    for params in grid:
        args = dict(model_name=f'train_stl_{params["subtraject_len"]}',
                    agent_class=QLearning, env_name=env_name,
                    env_big=env_big, **params)
        model = HERModel(**args)
        # model.train_runs(runs=runs)
        episodes = (model.env.height * model.env.width) ** 2 // 10
        model.train(episodes=episodes, max_episode_steps=150000)
        model.save_agent(f'{model.path}.agent.npy')
        model.save_plot_data(f'{model.path}.plot.npy')
        model.save_plot(f'{model.path}.plot')
