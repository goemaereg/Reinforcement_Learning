import gym
import gym_additions
import json
from tabular.agents import *
from tabular.hierarchical import *
from utils import save_plot
import random
import numpy as np
random.seed(0)
np.random.seed(0)

env_name = 'FourRooms-v0'
env = gym.make(env_name)
env_name = env_name[:-3]
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = {
    'env_shapes': shapes,
    'explo_horizon': 1,
    'explo_steps': 10,
    'learn_rate': 0.1,
    'gamma': 0.9,
    'lambda': 0.9,
    'n': 10
}

agent = ExploreOption(**d)

def single_run(agent, env, n_episodes, n_steps):
    evaluations_history = []
    agent.reset()
    # Training phase
    rewards_history = np.empty(n_episodes)
    for ep in range(n_episodes):
        obs = env.reset()
        cumreward = 0
        for step in range(n_steps):
            action = agent.act(obs)
            old_obs = obs # tuples don't have the copy problem
            obs, reward, done, info = env.step(action)
            cumreward += reward
            agent.learn(old_obs, action, reward, obs, done)
            if done:
                break
        rewards_history[ep] = cumreward
    env.close()
    return rewards_history

def multiple_runs(agent, env, n_runs, n_episodes, n_steps):
    perf = np.empty((n_runs, n_episodes))
    for run in range(n_runs):
        if (run%(n_runs//5)==0):
            print("\tRun {}/{}".format(run, n_runs))
        perf[run] = single_run(agent, env, n_episodes, n_steps)

    return perf.mean(axis=0)

n_episodes = 1500
n_steps = 150000 # virually never
n_runs = 100

perf = multiple_runs(agent, env, n_runs, n_episodes, n_steps)
# plotting
launch_specs = 'baseline'
file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
suptitle = "Performance of {} on {}{}".format(agent.name, env_name, env.roomsize)
title = agent.tell_specs()
xlabel = 'Episode'
ylabel = "Performance at {}".format(env_name)
save_plot(perf, file_name, suptitle, title, xlabel, ylabel,
          smooth_avg=n_episodes//100, only_avg=False)
