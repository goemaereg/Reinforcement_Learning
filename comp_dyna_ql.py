import gym
import gym_additions
import json
from tabular.agents import *
from tabular.hierarchical import *
from utils import save_plot

env_name = 'BlockingMaze-v0'
env = gym.make(env_name)
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = {
    'env_shapes': shapes,
    'explo_horizon': 1,
    'learn_rate': 0.05,
    'gamma': 0.9,
    'n': 10,
    'kappa': 0.01
}
agent = DynaQPlus(**d)

def single_run(env, agent, n_steps, n_episodes):
    # Training phase
    steps_history = np.empty(n_episodes)
    for ep in range(n_episodes):
        if (ep%(n_episodes//5)==0):
            print("\tEpisode {}/{}".format(ep, n_episodes))
            print("\t\tEpsilon: {}".format(agent.epsilon))
    #        evaluations_history.append(test(agent, env, n_steps,n_episodes=1))
    #        print("Task success rates: {}".format(agent.success_counter/agent.asked_counter))
    #        print("Current meta policy: {}".format(agent.meta_policy()))

        obs = env.reset()
        for step in range(n_steps):
            action = agent.act(obs)
            old_obs = obs # tuples don't have the copy problem
            obs, reward, done, info = env.step(action)
            agent.learn(old_obs, action, reward, obs, done)
            if done:
                break
        steps_history[ep] = step

    env.close()
    return steps_history

def avg_perf(env,agent, n_steps, n_episodes, n_runs):
    np.random.seed(0)
    random.seed(0)

    perf = np.empty((n_runs, n_episodes))
    for run in range(n_runs):
        print("Run {}/{}".format(run+1, n_runs))
        agent.reset()
        perf[run] = single_run(env, agent, n_steps, n_episodes)

    return perf.mean(axis=0)[1:]


def n_study(env, agent, n_spectrum, n_steps, n_episodes, n_runs):
    perfs = []
    for n in ns_spectrum:
        agent.n = n
        perfs.append(avg_perf(env, agent, n_steps, n_episodes, n_runs))
    return perfs

n_steps = 150000 # virually never
n_episodes = 3000
n_runs = 5
# ns_spectrum = [0, 5, 10, 50]

perfs = [
    avg_perf(env, DynaQ(**d),       n_steps, n_episodes, n_runs),
    avg_perf(env, DynaQPlus(**d),   n_steps, n_episodes, n_runs)
    #avg_perf(env, DynaQPlus2(**d),  n_steps, n_episodes, n_runs)
]
# plotting
#launch_specs = 'n{}'.format(agent.n)
launch_specs = 'dyna_comp'
file_name = "tabular/perf_plots/{}/{}".format(env_name, launch_specs)
title = "Episode length comparison on {} during training".format(env_name)
title += "\n{}".format(agent.tell_specs()) # last agent is good enough
xlabel = 'Episode'
ylabel = "Epsiode length (steps)"
save_plot(np.array(perfs), file_name, title, xlabel, ylabel,
          smooth_avg=n_episodes//100, only_avg=False, ylineat=1000,
          labels=['DynaQ', 'DynaQPlus'])
