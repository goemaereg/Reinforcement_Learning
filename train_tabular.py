import gym
import gym_additions
import json
from tabular.agents import *
from tabular.hierarchical import *
from utils import save_plot

env_name = 'FourRooms-v0'
env = gym.make(env_name)
env_name = env_name[:-3]
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = {
    'env_shapes': shapes,
    'explo_horizon': 1,
    'learn_rate': 0.05,
    'gamma': 1.,
    'lambda': 0.9,
    'n': 10
}

agent = EligibilityTraces(**d)

def test(agent, env, n_steps, n_episodes=10):
    #print("maxes of Qtable after taining: \n{}".format(np.max(agent.Qtable,axis=-1)))
    # Testing phase
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
            print("Action is {}".format(action))
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

#agent.play_around(env, n_steps=20000)

n_episodes = 500
n_steps = 150000 # virually never
evaluations_history = []
# Training phase
rewards_history = np.empty(n_episodes)
for ep in range(n_episodes):
    if (ep%(n_episodes//5)==0):
        print("Episode {}/{}".format(ep+1, n_episodes))
#        evaluations_history.append(test(agent, env, n_steps,n_episodes=1))
#        print("Task success rates: {}".format(agent.success_counter/agent.asked_counter))
#        print("Current meta policy: {}".format(agent.meta_policy()))

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
print("Final performance: {}".format(rewards_history[-1]))
#print("Task success rates: {}".format(agent.success_counter/agent.asked_counter))
#print("Current meta policy: {}".format(agent.meta_policy()))

# plotting
launch_specs = 'baseline'
file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
suptitle = "Performance of {} on {}".format(agent.name, env_name)
title = agent.tell_specs()
xlabel = 'Episode'
ylabel = "Performance at {}".format(env_name)
save_plot(rewards_history, file_name, suptitle, title, xlabel, ylabel,
          smooth_avg=n_episodes//100, only_avg=False)
