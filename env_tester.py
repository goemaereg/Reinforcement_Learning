import gym
import gym_additions
import json
from agents_core import Random_Agent

env = gym.make('RaceTrack-v0')
agent = Random_Agent((env.observation_space.shape, env.action_space.n))
print("Environment shapes:{}".format((env.observation_space.shape, env.action_space.n)))
n_episodes = 1
n_steps = 20

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
            print("Episode finished after {} timesteps"
                  " with cumulated reward {}".format(step+1, cumreward))
            env.render()
            break
env.close()
