import gym
import gym_additions
import json
from tabular.agents import *
from utils import *
import random
import numpy as np
random.seed(0)
np.random.seed(0)

env_name = 'FourRoomsMin-v0'
env = gym.make(env_name)
env_name = env_name[:-3]
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = {
    'env_shapes': shapes,
    'explo_horizon': 1,
    'explo_steps': 40,
    'learn_rate': 0.1,
    'gamma': 1.,
    'lambda': 0.9,
    'n': 10
}

agent = ExploreOption(**d)

def test(agent, env, n_episodes, n_steps):
    agent.reset()
    old_eps = agent.epsilon
    agent.epsilon = 0
    success_history = np.empty(n_episodes)
    for ep in range(n_episodes):
        obs = env.reset()
        cumreward = 0
        for step in range(n_steps):
            env.render()
            action = agent.act(obs)
            old_obs = obs # tuples don't have the copy problem
            obs, reward, done, info = env.step(action)
            print("Action {}".format(action))
            cumreward += reward
            if done:
                env.render()
                break
        success_history[ep] = int(cumreward == 100) # reached the max
    env.close()
    print("Finished with reward {}".format(cumreward))
    agent.epsilon = old_eps
    return success_history

def single_run(agent, env, n_episodes, n_steps):
    agent.reset()
    # Training phase
    success_history = np.empty(n_episodes)
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
        success_history[ep] = int(cumreward == 100) # reached the max
    env.close()
    return success_history

def multiple_runs(agent, env, n_runs, n_episodes, n_steps):
    perf = np.empty((n_runs, n_episodes))
    for run in range(n_runs):
        if (run%(n_runs//5)==0):
            print("\tRun {}/{}".format(run, n_runs))
        perf[run] = single_run(agent, env, n_episodes, n_steps)

    return perf.mean(axis=0)

def run_spectrum(agent, env, explo_spectrum, n_runs, n_episodes, n_steps):
    perfs = []
    for explo_steps in explo_spectrum:
        random.seed(0)
        np.random.seed(0)
        print("explo_step = {}".format(explo_steps))
        agent.explo_steps = explo_steps
        perf = multiple_runs(agent, env, n_runs, n_episodes, n_steps)
        perfs.append(smooth(perf, n_episodes//100))
    return np.array(perfs)

n_episodes = 2000
n_steps = 150000 # virually never
n_runs = 20
explo_spectrum = [20, 40, 60, 80]
#perf = multiple_runs(agent, env, n_runs, n_episodes, n_steps)
perf = run_spectrum(agent, env, explo_spectrum, n_runs, n_episodes, n_steps)

#test(agent, env, n_episodes=1, n_steps=n_steps)

# plotting
launch_specs = 'rs{}steps_spectrum_longer'.format(env.roomsize)
file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, classname(agent), launch_specs)
suptitle = "Success proportion of {} on {}".format(classname(agent), env_name)
title = agent.tell_specs()
xlabel = 'Episode'
ylabel = "Proportion of Global Goal reached".format(env_name)
#save_plot(perf, file_name, suptitle, title, xlabel, ylabel,
#          smooth_avg=n_episodes//100, only_avg=False, interval_yaxis=(0,1))

save_plot(perf, file_name, suptitle, title, xlabel, ylabel, interval_yaxis=(0,1),
        labels=['steps={}'.format(explo_steps) for explo_steps in explo_spectrum])
