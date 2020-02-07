import gym
import gym_additions
import json
from tabular.agents import *
from utils import *
import random
import numpy as np
from copy import deepcopy

## FUNCTIONS -------------------------------------------------------------------


def single_run(agent, env, n_steps):
    agent.reset()
    env.reset()
    # Training phase
    reward_history = np.empty(n_steps)
    obs = env.small_reset()
    cumreward = 0
    for step in range(n_steps):
        if (step%(n_steps//5)==0):
            print("\t\tStep {}/{}".format(step, n_steps))
        action = agent.act(obs)
        old_obs = obs # tuples don't have the copy problem
        obs, reward, done, info = env.step(action)
        cumreward += reward
        # reward = 0 # no reward (visualize explorer)
        agent.learn(old_obs, action, reward, obs, done)
        if (step==env.appear_obstacle) and isinstance(agent, QLearning):
            #agent.reset_eps()
            pass
        if done:
            obs = env.small_reset()

        reward_history[step] = cumreward
    env.close()
    return reward_history

def multiple_runs(agent, env, n_runs, n_steps):
    perf = np.empty((n_runs, n_steps))
    for run in range(n_runs):
        if (run%(n_runs//5)==0):
            print("\tRun {}/{}".format(run, n_runs))
        perf[run] = single_run(agent, env, n_steps)

    return perf.mean(axis=0)

def run_spectrum(agent, env, explo_spectrum, n_runs, n_steps, smoothed=False):
    perfs = []
    for explo_steps in explo_spectrum:
        print("explo_step = {}".format(explo_steps))
        agent.explo_steps = explo_steps
        perf = multiple_runs(agent, env, n_runs, n_steps)
#        if smoothed:
#            perf = smooth(perf, n_episodes//100)
        perfs.append(perf)

    return np.array(perfs)

def multiple_agents(agents, env, n_runs, n_steps):
    perfs = []
    for agent in agents:
        print("Agent: {}".format(agent.name))
        perfs.append(multiple_runs(agent, env, n_runs, n_steps))

    return np.array(perfs)

def print_policy(Q_agent, env):
    """ Prints the policy for all states of the environment. """
    policy = np.empty((env.height,env.width), dtype='str')#, dtype=np.object)
    Q_maxes = np.empty((env.height,env.width))
    for i in range(env.height):
        for j in range(env.width):
            s = (i,j)
            Q_s = Q_agent.Qtable[s]
            # print(i,j,max(Q_s))
            Q_maxes[s] = max(Q_s)
            best_a = allmax(Q_s)[0]
            #policy[i,j] = [env.moves_str[a] for a in best_a]
            policy[i,j] = env.moves_str[best_a] if max(Q_s) != 0 else '0'
    print("Policy: \n{}".format(policy))
    print("Q maxes: \n{}".format(Q_maxes))
    return policy, Q_maxes

def draw_optimal_perf(n_steps, env):
    slope_before = 1/10
    slope_after = 1/16
    if envname(env) == 'ShortcutMaze':
        slope_after, slope_before = slope_before, slope_after
    optimal_perf_before = np.arange(env.appear_obstacle)*slope_before
    optimal_perf_after = np.arange(n_steps-env.appear_obstacle)*slope_after
    optimal_perf_after += optimal_perf_before[-1]
    optimal_perf = np.concatenate([optimal_perf_before, optimal_perf_after])
    return optimal_perf

## HYPERPARAMETERS & PREPARATION -----------------------------------------------
env_name = 'ShortcutMaze-v0'
env = gym.make(env_name)
env_name = env_name[:-3]
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = { # inputs for the agent
    'env_shapes': shapes,
    'explo_horizon': 5000,
    'min_eps': 0.1,
    'explo_steps': 15,
    'learn_rate': 0.2,
    'gamma': 0.99,
    'lbda': 0.99,
    'beta': 1,
    'n': 10,
    'exploiter_class': QLearning,
    'explorer_class': QLearning,
}

n_episodes = 3
n_steps = 50000
n_runs = 10

agent = QLearning_Optimistic(**d)
explo_spectrum = [1,3,5,7,10,15,20,30,60,3600]
agents = [
    ExploreOption(**d),
    TreeBackup(**d),
    EligibilityTraces(**d),
    QLearning(**d),
]
optimal_perf = draw_optimal_perf(n_steps, env)

## LAUNCH TRAINING -------------------------------------------------------------
perf = multiple_runs(agent, env, n_runs, n_steps)
#perf = run_spectrum(agent, env, explo_spectrum, n_runs, n_steps, smoothed=False)
#perf = multiple_agents(agents, env, n_runs, n_episodes, n_steps)

perf = np.vstack([optimal_perf, perf]) # adding perfect perf comparison

## PRINTING STUFF --------------------------------------------------------------
if isinstance(agent, ExploreOption):
    print("Exploiter:")
    policy, Q_maxes = print_policy(agent.exploiter, env)
    print("Explorer:")
    policy, Q_maxes = print_policy(agent.explorer, env)
    print("Visit counts:")
    print(agent.visit_counts.sum(axis=-1))
else:
    policy, Q_maxes = print_policy(agent, env)

## PLOTTING --------------------------------------------------------------------
launch_specs = '{}_baseline'.format(agent.short_name) # name of output plot file
#file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, classname(agent), launch_specs)
file_name = "tabular/perf_plots/{}/{}".format(env_name, launch_specs)
suptitle = "{} on {}".format(classname(agent), env_name)
#suptitle = "Agents comparison on {}".format(env_name)
title = agent.tell_specs()
xlabel = 'Time steps'
ylabel = "Cumulative Reward".format(env_name)
#labels = ['Optimal', *['c={}'.format(explo_step) for explo_step in explo_spectrum]]
labels = ['Optimal', agent.short_name]
save_plot(perf, file_name, suptitle, title, xlabel, ylabel, ylineat=env.appear_obstacle,
          smooth_avg=0, only_avg=False, labels=labels)
