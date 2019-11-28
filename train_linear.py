import gym
import gym_additions
import json
from linear.agents_tiling import *
from utils import save_plot

env_name = 'Acrobot-v2'
env = gym.make(env_name)
print(env.observation_space)
shapes = ((env.observation_space.n,), env.action_space.n)
n_episodes = 5000
n_steps = 500
d = {
    'env_shapes': shapes,
    'explo_horizon': n_steps*n_episodes/5,
    'min_eps': 0.1,
    'learn_rate': 0.01,
    'gamma':.9,
    'n': 10
}
agent = QLearning(**d)

def test(agent, env, n_steps, n_episodes=10):
    agent.verbose = True
    old_eps = agent.epsilon
    agent.epsilon = 0
    rewards_history = np.empty(n_episodes)
    for ep in range(n_episodes): # everything is deterministic
        obs = env.reset()
        cumreward = 0
        for step in range(n_steps):
            env.render()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            cumreward += reward
            if done:
                print("Episode finished after {} timesteps"
                      " with cumulated reward {}".format(step+1, cumreward))
                env.render()
                break
        rewards_history[ep] = cumreward
    env.close()
    agent.verbose = False
    agent.epsilon = old_eps
    return rewards_history.mean()

evaluations_history = []
# Training phase
rewards_history = np.empty(n_episodes)
for ep in range(n_episodes):
    if (ep%(n_episodes//5)==0):
        print("Episode {}/{}".format(ep+1, n_episodes))
        evaluations_history.append(test(agent, env, n_steps, n_episodes=3))

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
    if np.random.rand()<0.01: print("Step {}".format(step))
    rewards_history[ep] = cumreward

env.close()
print("Evaluations history: {}".format(evaluations_history))

#print("Task success rates: {}".format(agent.success_counter/agent.asked_counter))
#print("Current meta policy: {}".format(agent.meta_policy()))

# plotting
env_name = env_name[:-3]
launch_specs = 'baseline'
file_name = "linear/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
suptitle = "Performance of {} on {}".format(agent.name, env_name)
title = agent.tell_specs()
xlabel = 'Episode'
ylabel = "Performance at {}".format(env_name)
save_plot(rewards_history, file_name, suptitle, title, xlabel, ylabel,
          smooth_avg=n_episodes//100, only_avg=False)
