import gym
import gym_additions
import json
from agents_core import Random_Agent
import numpy as np
np.random.seed(0)
import random
random.seed(0)

env = gym.make('ShortCorridor-v0')
obs_shape = None #tuple([s.n for s in env.observation_space])
agent = Random_Agent((obs_shape, env.action_space.n))
n_episodes = 1
n_steps = 200
rewards_history = np.empty(n_episodes)
for ep in range(n_episodes):
    obs = env.reset()
    cumreward = 0
    for step in range(n_steps):
        env.render()
        action = agent.act(obs)
        print("\tAction is {} in state {}".format(action, obs))
        obs, reward, done, info = env.step(action)
        cumreward += reward
        if done:
            env.render()
            print("Episode finished after {} timesteps"
                  " with cumulated reward {}".format(step+1, cumreward))
            break
    rewards_history[ep] = cumreward
env.close()

print("Average reward: {}".format(rewards_history.mean()))
