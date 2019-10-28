import numpy as np
from utils import save_plot
from bandit.core import *
from bandit.agents import *
import time

def run(bandit, agent, max_steps):
    """ Single run of a bandit  problem (Bandit-Agent interactions) """
    reward_record = np.empty(max_steps)
    agent.reset()
    bandit.reset()
    action_time_record = np.empty(max_steps)
    learn_time_record = np.empty(max_steps)
    for step in range(max_steps):
        init = time.time()
        action = agent.act()
        action_time_record[step] = time.time() - init

        reward = bandit.pull(action)

        init = time.time()
        agent.learn(action, reward)
        learn_time_record[step] = time.time() - init
        reward_record[step] = reward

    if np.random.rand() < 0.01:
        print("\t\tAverage action time: {}".format(action_time_record.mean()))
        print("\t\tAverage learn time: {}".format(learn_time_record.mean()))
    return reward_record

def many_runs(bandit, agent, n_runs, max_steps=1000, plotting=True):
    """ Runs n_runs of a given bandit problem, each returning the online rewards
        We plot their average over the runs if 'plotting'.
        """
    reward_record_runs = np.empty((n_runs, max_steps))
    for r in range(n_runs):
        if (r%(n_runs//5)==0): print("\tRun {}/{}".format(r, n_runs))
        reward_record_runs[r] = run(bandit, agent, max_steps)

    perf = np.mean(reward_record_runs, axis=0)
    if plotting:
        save_plot(perf, 'bandit/plots/{}'.format(agent.name), "{} performance average over {} runs\nParameters: {}".format(agent.name, n_runs, agent.tell_specs()), xlabel='Step', ylabel='Reward')

    return perf


#print("|Q-q| distance: {}".format(np.linalg.norm(agent.Qs - bandit.reveal_qstar())))
if __name__ == '__main__':
    n_arms = 10
    bandit = Bandit(lever_types=[Gaussian_Lever]*n_arms)
    agents = [EpsGreedy(n_arms, epsilon=1/16),
              NonStat_EpsGreedy(n_arms, alpha=1/16),
              OptimisticInits(n_arms, Q0=1.),
              Gradient_Bandit(n_arms, alpha=1/4),
              UCB(n_arms, 1.)]
    n_runs = 1000
    max_steps = 1000
    perfs = np.empty((len(agents), max_steps))
    for i,agent in enumerate(agents):
        print("Agent {}".format(agent.name))
        agent_init = time.time()
        perfs[i] = many_runs(bandit, agent, n_runs, max_steps, plotting=False)
        print("\tDone in {}".format(time.time() - agent_init))

    save_plot(perfs, 'bandit/plots/all_agents',
              "Performance average over {} runs of all agents".format(n_runs),
              xlabel='Step', ylabel='Reward',
              labels=[agent.name for agent in agents])
