import gym
import gym_additions
import json
from agents_core import Random_Agent
import numpy as np
np.random.seed(0)
import random
random.seed(0)

env = gym.make('DynaMaze-v0')
obs_shape = tuple([s.n for s in env.observation_space])
agent = Random_Agent((obs_shape, env.action_space.n))
n_episodes = 10
n_steps = 2000

for ep in range(n_episodes):
    obs = env.reset()
    cumreward = 0
    for step in range(n_steps):
        env.render()
        action = agent.act(obs)
        print("Action is {}".format(action))
        obs, reward, done, info = env.step(action)
        cumreward += reward
        if done:
            env.render()
            print("Episode finished after {} timesteps"
                  " with cumulated reward {}".format(step+1, cumreward))
            break
env.close()
