import gym
import gym_additions
import json
from agents_core import Random_Agent

env = gym.make('TicTacToe-v0')
agent = Random_Agent((env.observation_space.shape, env.action_space.n))

n_episodes = 10
n_steps = 10

for ep in range(n_episodes):
    obs = env.reset()
    for step in range(n_steps):
        env.render()
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            env.render()
            break
env.close()
