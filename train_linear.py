import gym
import gym_additions
import json
from linear.agents import *
from linear.evolutionary import *
from utils import save_plot, ensure_dir
import random

np.random.seed(0)
random.seed(0)

env_name = 'MountainCar-v2'
env = gym.make(env_name)
version = env_name[-2:]
env_name = env_name[:-3]
print(env.observation_space)
if int(version[-1])<=3:
    shapes = ((env.observation_space.n,), env.action_space.n)
else:
    shapes = ((env.observation_space.spaces[0].n,env.observation_space.spaces[1].n), env.action_space.n)
#shapes = ((env.observation_space.shape[0],), env.action_space.n)
n_episodes = 5000
n_steps = 200
d = {
    # classic RL
    'env_shapes': shapes,
    'explo_horizon': 1,
    'min_eps': 0.1,
    'learn_rate': 0.001,
    'learn_rate_w': 0.001,
    'temperature': 1,
    'gamma':1,
    'lmbda': 0.9,
    'n': 10,
    # evolutionary algorithms
    'N': 50,
    'std': 0.05,
    'mu': 20
}
agent = QLearning(**d)

def test(agent, env, n_steps, n_episodes=10):
    agent.verbose = True
#    old_eps = agent.epsilon
#    agent.epsilon = 0
    rewards_history = np.empty(n_episodes)
    for ep in range(n_episodes): # everything is deterministic
        obs = env.reset()
        cumreward = 0
        for step in range(n_steps):
            env.render()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if np.random.rand() < 0.01:  print("Observation: {}".format(obs))
            cumreward += reward
            if done:
                print("Episode finished after {} timesteps"
                      " with cumulated reward {}".format(step+1, cumreward))
                env.render()
                break
        rewards_history[ep] = cumreward
    env.close()
    agent.verbose = False
#    agent.epsilon = old_eps
    return rewards_history.mean()

evaluations_history = []
# Training phase
rewards_history = np.empty(n_episodes)
for ep in range(n_episodes):
    if (ep%(n_episodes//5)==0):
        print("Episode {}/{}".format(ep+1, n_episodes))
        evaluations_history.append(test(agent, env, n_steps, n_episodes=1))

    obs = env.reset()
    cumreward = 0
    for step in range(n_steps):
        action = agent.act(obs)
        old_obs = obs
        obs, reward, done, info = env.step(action)
        done_time = (step == n_steps-1)
        cumreward += reward
        agent.learn(old_obs, action, reward, obs, d=(done or done_time))
        if done:
            break
    if np.random.rand()<0.01: print("Step {}".format(step))
    rewards_history[ep] = cumreward

env.close()
print("Evaluations history: {}".format(evaluations_history))

# plotting
launch_specs = 'repro_baseline'+version
file_name = "linear/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
ensure_dir(file_name)
suptitle = "Performance of {} on {}".format(agent.name, env_name)
title = agent.tell_specs()
xlabel = 'Episode'
ylabel = "Performance at {}".format(env_name)
save_plot(rewards_history, file_name, suptitle, title, xlabel, ylabel,
          smooth_avg=n_episodes//100, only_avg=False)
