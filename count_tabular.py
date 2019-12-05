import gym
import gym_additions
import json
from tabular.agents import *
from tabular.hierarchical import *
from utils import save_plot
np.random.seed(0)
random.seed(0)

env_name = 'FourRooms-v0'
env = gym.make(env_name)
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = {
    'env_shapes': shapes,
    'explo_horizon': 1,
    'learn_rate': 0.05,
    'gamma': 0.9,
    'lambda': 0.9,
    'n': 10
}
agent = EligibilityTraces(**d)

#agent.play_around(env, n_steps=20000)

n_episodes = 3000
n_steps = 150000 # virually never
evaluations_history = []
# Training phase
steps_history = np.empty(n_episodes)
for ep in range(n_episodes):
    if (ep%(n_episodes//5)==0):
        print("Episode {}/{}".format(ep+1, n_episodes))

    obs = env.reset()
    cumreward = 0
    for step in range(n_steps):
        action = agent.act(obs)
        old_obs = obs # tuples don't have the copy problem
        obs, reward, done, info = env.step(action)
        agent.learn(old_obs, action, reward, obs, done)
        if done:
            break
    if np.random.rand()<0.01: print("Step {}".format(step))
    steps_history[ep] = step

env.close()
steps_history = steps_history[1:]
print("Final performance: {}".format(steps_history[-1]))

# plotting
launch_specs = 'baseline'
file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
suptitle = "Steps to goal of {} on {}".format(agent.name, env_name[:-3])
title = agent.tell_specs()
xlabel = 'Episode'
ylabel = "Performance at {}".format(env_name)
save_plot(steps_history, file_name, suptitle, title, xlabel, ylabel,
          smooth_avg=n_episodes//100, only_avg=False, xlineat=steps_history[-1])
