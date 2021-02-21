from tabular.agents import QLearning
from sklearn.model_selection import ParameterGrid
from model import HERModel


# env_big = True
env_big = False
env_name = 'FourRoomsGoalBig-v0' if env_big else 'FourRoomsGoal-v0'

# grid = ParameterGrid({'subtraject_len': range(4, 64, 8)})
grid = ParameterGrid({'subtraject_len': range(28, 30, 8)})

if __name__ == '__main__':
    runs=10
    for params in grid:
        args = dict(model_name=f'train_stl_{params["subtraject_len"]}',
                    agent_class=QLearning, env_name=env_name,
                    env_big=env_big, **params)
        model = HERModel(**args)
        print(model.__dict__)
        model.train_runs(runs=runs)
        model.save_agent(f'{model.path}.agent.npy')
        model.save_plot_data(f'{model.path}.plot.npy')
        model.save_plot(f'{model.path}.plot')
