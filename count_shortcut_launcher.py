import gym
import gym_additions
import json
from tabular.agents import *
from tabular.fancy_agents import *
from tabular.intrinsic_reward_functions import *
from utils import *
import random
import numpy as np
from copy import deepcopy
np.set_printoptions(precision=5, suppress=True)

## FUNCTIONS -------------------------------------------------------------------
def test_agent(agent, env, n_episodes=3, max_steps=100):
    """ The agent is tested for n_episodes, returning average perf.
    No epsilon, no learning.
    We measure how long the agents takes to reach the goal."""
    # Testing phase
    env_copy = deepcopy(env) # not to alter current episode in training
    if isinstance(agent, ExploreOption) or isinstance(agent, ExploreAndExploit):
        agent.test_mode = True
    if hasattr(agent, 'epsilon'):
        old_eps = agent.epsilon
        agent.epsilon = 0 #1e-3
    steps_history = np.empty(n_episodes)
    for ep in range(n_episodes):
        obs = env_copy.small_reset()
        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env_copy.step(action)
            if done:
                break

        steps_history[ep] = step+1 # step starts at 0
    env.close()
    if hasattr(agent, 'epsilon'):
        agent.epsilon = old_eps
    if isinstance(agent, ExploreOption) or isinstance(agent, ExploreAndExploit):
        agent.test_mode = False
    return steps_history.mean()

def single_run(agent, env, n_steps, test_every=100):
    agent.reset()
    env.reset()
    ## Training phase
    steps_history = []
    obs = env.small_reset()
    cumreward = 0
    for step in range(n_steps):
        ## Printing and testing
        if (step%(n_steps//5)==0):
            print("\t\tStep {}/{}".format(step, n_steps))
            pass
        if (step%test_every == 0):
            perf = test_agent(agent, env)
            steps_history.append(perf)
        if step in (env.appear_obstacle, n_steps-1):
            wrapper_print_policy(agent, env)

        ## Actual loop --------
        action = agent.act(obs)
        old_obs = obs # tuples don't have the copy problem
        obs, reward, done, info = env.step(action)
        agent.learn(old_obs, action, reward, obs, done)
        if (step==env.appear_obstacle) and isinstance(agent, QLearning):
            agent.reset_eps() # now the "if" convers just QLearning
            pass
        if done:
            obs = env.small_reset()

    env.close()
    return np.array(steps_history)

def multiple_runs(agent, env, n_runs, n_steps, test_every=100):
    perf = np.empty((n_runs, n_steps//test_every))
    for run in range(n_runs):
        if (n_runs > 5) and (run%(n_runs//5)==0):
            print("\tRun {}/{}".format(run, n_runs))
        perf[run] = single_run(agent, env, n_steps, test_every)

    return perf.mean(axis=0)

def run_spectrum(agent, env, spectrum, n_runs, n_steps,
                 test_every=100, smoothed=False):
    """Takes as input a spectrum
        (string name of the variable to change,
         [values the variable should take])
        Runs multiple_runs for each of these values"""
    perfs = []
    varname, values = spectrum
    for value in values:
        print("{} = {}".format(varname, value))
        setattr(agent, varname, value)
        perf = multiple_runs(agent, env, n_runs, n_steps, test_every)
#        if smoothed:
#            perf = smooth(perf, n_episodes//100)
        perfs.append(perf)

    return np.array(perfs)

def multiple_agents(agents, env, n_runs, n_steps, test_every=100):
    perfs = []
    for agent in agents:
        print("Agent: {}".format(agent.name))
        perfs.append(multiple_runs(agent, env, n_runs, n_steps, test_every))

    return np.array(perfs)

def print_policy(Q_agent, env):
    """ Prints the policy for all states of the environment. """
    policy = np.empty((env.height,env.width), dtype='str')#, dtype=np.object)
    Q_maxes = np.empty((env.height,env.width))
    for i in range(env.height):
        for j in range(env.width):
            s = (i,j)
            Q_s = Q_agent.Qtable[s]
            Q_maxes[s] = max(Q_s)
            best_a = allmax(Q_s)[0]
            policy[i,j] = env.moves_str[best_a] if max(Q_s) != 0 else '0'
    print("Policy: \n{}".format(policy))
    print("Q maxes: \n{}".format(Q_maxes))
    return policy, Q_maxes

def wrapper_print_policy(agent, env):
    if isinstance(agent, ExploreOption) or isinstance(agent, ExploreAndExploit):
        print("Exploiter:")
        policy, Q_maxes = print_policy(agent.exploiter, env)
        if isinstance(agent.explorer, Sarsa):
            print("Explorer:")
            policy, Q_maxes = print_policy(agent.explorer, env)
        print("Visit counts:")
        if agent.visit_counts.ndim == 3:
            print(agent.visit_counts.sum(axis=-1))
        else:
            print(agent.visit_counts)
    else:
        policy, Q_maxes = print_policy(agent, env)
        if hasattr(agent, 'visit_counts'):
            print("Visit counts:")
            print(agent.visit_counts)

def draw_optimal_perf(n_steps, env, test_every):
    steps_before = 10 # optimal path before change
    steps_after = 16 # optimal path after change
    if envname(env) == 'ShortcutMaze':
        steps_after, steps_before = steps_before, steps_after
    optimal_perf_before = np.ones(env.appear_obstacle//test_every)*steps_before
    optimal_perf_after = np.ones((n_steps-env.appear_obstacle)//test_every)*steps_after
    optimal_perf = np.concatenate([optimal_perf_before, optimal_perf_after])
    return optimal_perf

## HYPERPARAMETERS & PREPARATION -----------------------------------------------
env_name = 'ShortcutMaze-v0'
env = gym.make(env_name)
env_name = env_name[:-3]
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = { # inputs for the agent; ignored if not concerned
    'env_shapes': shapes,   # environment shapes (input, output) for all algos
    'explo_horizon': 5000,  # steps until reaching minimal eps value in annealing
    'min_eps': 0.1,         # minimal eps value in annealing eps algos
    'min_eps_explo': 0.1,   # same but for the explorer
    'c_switch': 15,         # Explore algos, exploration steps
    'learn_rate': 0.1,      # most algos, learning rate
    'lrEO': 0.001,          # ExploreOption learning rate
    'gamma': 0.99,           # all algos, MDP gamma
    'gamma_explo': 0.9,      # Explore algos, explorer gamma
    'ex_prob': 0.5,         # probability to explore (Explore algos)
    'lbda': 0.99,           # lambda, traces algos
    'beta': 1.,             # R= r_e + beta*r_i intrinsic reward algos
    'kappa': 1e-8,          # DynaQ+ explo param
    'epsilon': 0.1,         # NOT the eps-greedy epsilon. For Delayed_QLearning.
    'delta': 0.1,           # Delayed_QLearning
    'm': 5,                 # Delayed_QLearning. Leave None for default (and failure)
    'eps1': 0.1,            # Delayed_QLearning. Leave None for default (and failure)
    'n': 10,                # Dyna algos model steps
    'exploiter_class': QLearning,           # Explore algos
    'explorer_class': QLearning_Optimistic, # Explore algos
    'reward_function': Inverse_sqrt,       # Explore algos
}

n_episodes = 10
test_every = 100 # periodicity of testing during training (in steps)
n_steps = 1000*100 # here 1000 is therefore the number of points on the curve
n_runs = 50

agent = QLearning_VC(**d)
#spectrum = ('c_switch', [1,3,5,7,10,15,20,30,60,3600])
#spectrum = ('c_switch', [7,10,15,20]) # short baseline
#spectrum = ('c_switch', [15]) # so I don't have to change shit
spectrum = ('beta', [0.1,0.5,0.75,1, 1.25, 1.5])
#spectrum = ('beta', [0.01,0.05,0.075,0.1, 0.125, 0.15])
#spectrum = ('beta', [0.001,0.005,0.0075,0.01, 0.0125, 0.015])
#spectrum = ('beta', [0.125])
#spectrum = ('ex_prob', [0.2,0.35,0.5,0.75,1])
#spectrum = ('min_eps', [0,0.01,0.1,0.2,0.3,0.5,1])
#spectrum = ('gamma_explo', [0.5,0.8,0.9,0.99,1])
#spectrum = ('lrEO', [0,0.001,0.005,0.01,0.02,0.05,0.1])
#spectrum = ('delta', [0.001,0.01,0.1])
#spectrum = ('eps1', [0.001,0.01,0.1,0.5])
#spectrum = ('m', [5,10,25,50]) # short baseline
#spectrum = ('n', [1,2,5,10])
#spectrum = ('kappa', [1e-7, 5e-7, 1e-6])

agents = [
    ExploreOption(**d),
    TreeBackup(**d),
    EligibilityTraces(**d),
    QLearning(**d),
]
optimal_perf = draw_optimal_perf(n_steps, env, test_every)

## LAUNCH TRAINING -------------------------------------------------------------
#perf = multiple_runs(agent, env, n_runs, n_steps, test_every)
perf = run_spectrum(agent, env, spectrum, n_runs, n_steps, test_every, smoothed=False)
#perf = multiple_agents(agents, env, n_runs, n_episodes, n_steps, test_every)

perf = np.vstack([optimal_perf, perf]) # adding perfect perf comparison

## PRINTING STUFF --------------------------------------------------------------
wrapper_print_policy(agent, env)
print(test_agent(agent, env, n_episodes, n_steps))

## PLOTTING --------------------------------------------------------------------
launch_specs = '{}_beta_g.99_2'.format(agent.short_name) # name of output plot file
#file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, classname(agent), launch_specs)
file_name = "tabular/perf_plots/{}/{}".format(env_name, launch_specs)
suptitle = "{} on {}".format(classname(agent), env_name)
#suptitle = "Agents comparison on {}".format(env_name)
title = agent.tell_specs()
xlabel = '{} Time steps'.format(test_every)
ylabel = "Steps to goal".format(env_name)
labels = ['Optimal']
#labels += [agent.name]
labels += ['{}={}'.format(spectrum[0], value) for value in spectrum[1]]
save_plot(perf, file_name, suptitle, title, xlabel, ylabel, ylineat=env.appear_obstacle//test_every,
          smooth_avg=0, only_avg=False, labels=labels)
