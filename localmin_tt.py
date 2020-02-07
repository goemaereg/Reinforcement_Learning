import gym
import gym_additions
import json
from tabular.agents import *
from utils import *
import random
import numpy as np
# random.seed(0)
# np.random.seed(0)

## Functions
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
        success_history[ep] = int(cumreward == 10) # reached the max
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
        success_history[ep] = int(cumreward == 10) # reached the max
    env.close()
    return success_history

def multiple_runs(agent, env, n_runs, n_episodes, n_steps):
    perf = np.empty((n_runs, n_episodes))
    for run in range(n_runs):
        if (run%(n_runs//5)==0):
            print("\tRun {}/{}".format(run, n_runs))
        perf[run] = single_run(agent, env, n_episodes, n_steps)

    return perf.mean(axis=0)

def run_spectrum(agent, env, explo_spectrum, n_runs, n_episodes, n_steps, smoothed=False):
    perfs = []
    for explo_steps in explo_spectrum:
        print("explo_step = {}".format(explo_steps))
        agent.explo_steps = explo_steps
        perf = multiple_runs(agent, env, n_runs, n_episodes, n_steps)
        if smoothed:
            perf = smooth(perf, n_episodes//100)
        perfs.append(perf)

    return np.array(perfs)


def multiple_agents(agents, env, n_runs, n_episodes, n_steps):
    perfs = []
    for agent in agents:
        print("Agent: {}".format(agent.name))
        perfs.append(multiple_runs(agent, env, n_runs, n_episodes, n_steps))

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
            policy[i,j] = env.moves_str[best_a]
    print("Policy: \n{}".format(policy))
    print("Q maxes: \n{}".format(Q_maxes))
    return policy, Q_maxes

## Hyperparameters
env_name = 'LocalMin-v0'
env = gym.make(env_name)
env_name = env_name[:-3]
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = { # inputs for the agent
    'env_shapes': shapes,
    'explo_horizon': 1,
    'min_eps': 0.1,
    'min_eps_explo': 0.1,
    'explo_steps': 15,
    'learn_rate': 0.1,
    'gamma': 0.99,
    'lbda': 0.99,
    'n': 10,
    'exploiter_class': QLearning,
    'explorer_class': QLearning,
}

n_episodes = 2000
n_steps = 150000 # virually never
n_runs = 10

agent = ExploreOption(**d)
explo_spectrum = [1,3,5,7,10,15,20,30,60,3600]
agents = [
    ExploreOption(**d),
    TreeBackup(**d),
    EligibilityTraces(**d),
    QLearning(**d),
]
#perf = single_run(agent, env, n_episodes, n_steps)
perf = multiple_runs(agent, env, n_runs, n_episodes, n_steps)
#perf = run_spectrum(agent, env, explo_spectrum, n_runs, n_episodes, n_steps, smoothed=True)
#perf = multiple_agents(agents, env, n_runs, n_episodes, n_steps)

q_agent = agent.exploiter if isinstance(agent, ExploreOption) else agent
policy, Q_maxes = print_policy(q_agent, env)
# quit()
# plotting
launch_specs = '{}_baseline_repeat'.format(agent.short_name) # name of output plot file
#file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, classname(agent), launch_specs)
file_name = "tabular/perf_plots/{}/{}".format(env_name, launch_specs)
suptitle = "Success proportion of {} on {}".format(classname(agent), env_name)
#suptitle = "Agents comparison on {}".format(env_name)
title = agent.tell_specs()
xlabel = 'Episode'
ylabel = "Proportion of Global Goal reached".format(env_name)
#labels = [agent.name for agent in agents]
#labels=['steps={}'.format(explo_steps) for explo_steps in explo_spectrum]
labels = None
save_plot(perf, file_name, suptitle, title, xlabel, ylabel,
          smooth_avg=n_episodes//100, only_avg=False, interval_yaxis=(0,1), labels=labels)
