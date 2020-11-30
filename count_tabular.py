import gym
import gym_additions
import json
from tabular.agents import *
from tabular.hierarchical import *
from utils import save_plot

env_name = 'FourRoomsGoal-v0'
env = gym.make(env_name)
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = {
    'env_shapes': shapes,
    'explo_horizon': 1,
    'learn_rate': 0.05,
    'explo_steps': 10,
    'gamma': 0.9,
    'lambda': 0.9,
    'n': 10
}

n_episodes = 3000
n_steps = 150000 # virually never

def test_agent(agent, env, n_episodes, n_steps):
    """ Returns the steps_history of the agent"""
    evaluations_history = []
    # Training phase
    steps_history = np.empty(n_episodes)
    for ep in range(n_episodes):
        if (ep%(n_episodes//5)==0):
            print("Episode {}/{}".format(ep+1, n_episodes))
        obs = env.reset()
        for step in range(n_steps):
            action = agent.act(obs)
            old_obs = obs # tuples don't have the copy problem
            obs, reward, done, info = env.step(action)
            agent.learn(old_obs, action, reward, obs, done)
            if done:
                break
                if np.random.rand()<0.01: print("Step {}".format(step))

            steps_history[ep] = step
        if ep==0:
            print("First trial in {} steps".format(step))
    env.close()
    return steps_history[1:] # first is purely random

def smooth(perf, smooth_avg):
    perf_smooth = []
    perf_smooth += [np.mean(perf[i-smooth_avg:i+smooth_avg])
                    for i in range(smooth_avg, len(perf)-smooth_avg)]
    return perf_smooth

# agent = QLearning(**d)
agents = [
    #MonteCarlo(**d),
    #EligibilityTraces(**d),
    # TreeBackup(**d),
    QLearning(**d),
    # ExploreOption(**d),
    #Sarsa(**d)
    ]
if len(agents) == 1:
    agent = agents[0]
    perf = test_agent(agent, env, n_episodes, n_steps)
    print("Final performance: {}".format(perf[-1]))
    # plotting
    launch_specs = 'perf{}'.format(env.roomsize)
    file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
    suptitle = "{} performance on {}{}".format(agent.name, env_name[:-3], env.roomsize)
    title = agent.tell_specs()
    xlabel = 'Episode'
    ylabel = "Performance at {}".format(env_name)
    save_plot(perf, file_name, suptitle, title, xlabel, ylabel,
              smooth_avg=n_episodes//100, only_avg=True)

else:
    perfs = []
    for agent in agents:
        np.random.seed(0)
        random.seed(0)
        perf = test_agent(agent, env, n_episodes, n_steps)
        # The smoothing is done here as there are multiple curves
        perfs.append(smooth(perf, n_episodes//100))
        print("Final performance: {}".format(perf[-1]))

    perfs = np.array(perfs)
    print(perfs.shape)
    # plotting
    launch_specs = 'comp'
    #file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
    file_name = "tabular/perf_plots/{}/{}".format(env_name, launch_specs)
    suptitle = "Agent Performance Comparison on {}{}".format(env_name[:-3], env.roomsize)
    title = agent.tell_specs()
    xlabel = 'Episode'
    ylabel = "Performance at {}".format(env_name)
    save_plot(perfs, file_name, suptitle, title, xlabel, ylabel,
              smooth_avg=0, labels=[agent.name for agent in agents])
