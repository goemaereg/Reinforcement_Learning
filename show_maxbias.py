import gym
import gym_additions
import json
from tabular.agents import *
from utils import save_plot

env_name = 'MaxBias-v0'
env = gym.make(env_name)
agent = QLearning\
    (((env.observation_space.n,), env.action_space.n),
    learn_rate=0.1, explo_horizon=1, gamma=1)

def single_run(agent, env, n_steps, n_episodes):
    """ Runs a single learning phase of the MaxBias env;
    returns the actions used throughout. We know action 1 is best. """
    actions_A_history = np.empty(n_episodes) # we see state A once per episode
    for ep in range(n_episodes):
        obs = env.reset()
        for step in range(n_steps):
            action = agent.act(obs)
            old_obs = obs
            if obs == 1: # I know this is state A in the env
                actions_A_history[ep] = action%2
            obs, reward, done, info = env.step(action)
            agent.learn(old_obs, action, reward, obs, done)
            if done:
                break
    env.close()
    non0 = np.count_nonzero(actions_A_history)
    return actions_A_history

n_steps = 5 # episodes only last 2
n_episodes = 300
n_runs = 10000
# Training phase
actions_A_histories = np.empty((n_runs, n_episodes))
for run in range(n_runs):
    if run%(n_runs//5)==0: print("Run {}/{}".format(run, n_runs))
    agent.reset()
    actions_A_histories[run] = single_run(agent, env, n_steps, n_episodes)

print("Qtable: {}".format(agent.Qtable))
left_percent = 1 - actions_A_histories.mean(axis=0) #
# plotting
launch_specs = 'maxbias'
title = "Bad action proportion of {}; {} runs".format(agent.name, n_runs)
file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
title += "\n{}".format(agent.tell_specs())
xlabel = 'Episode'
ylabel = "Proportion of action LEFT from A".format(env_name)
save_plot(left_percent, file_name, title, xlabel, ylabel,
          smooth_avg=len(left_percent)//100, only_avg=False, force_xaxis_0=True, xlineat=0.025)
