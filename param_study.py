import numpy as np
from utils import save_plot
from bandit.core import *
from bandit.agents import *
from bandit_testbed import run, many_runs
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#print("|Q-q| distance: {}".format(np.linalg.norm(agent.Qs - bandit.reveal_qstar())))
if __name__ == '__main__':
    n_arms = 10
    bandit = Bandit(lever_types=[Gaussian_Lever]*n_arms)
    total_range = 2**np.arange(-7,3, dtype=float)
    epsilon_range = total_range[0:6]
    nonstat_range = total_range[2:7]
    optimistic_range = total_range[5:]
    grad_alpha_range = total_range[2:]
    ucb_range = total_range[3:]
    ranges = [epsilon_range, nonstat_range, optimistic_range, grad_alpha_range, ucb_range]

    eps_agents =    [EpsGreedy(n_arms, epsilon=epsilon)
                        for epsilon in epsilon_range]
    ns_eps_agents = [NonStat_EpsGreedy(n_arms, alpha=alpha)
                        for alpha   in nonstat_range]
    opt_agents =    [OptimisticInits(n_arms, Q0=Q0)
                        for Q0      in optimistic_range]
    grad_agents =   [Gradient_Bandit(n_arms, alpha=alpha)
                        for alpha   in grad_alpha_range]
    ucb_agents =    [UCB(n_arms, c=c)
                        for c       in ucb_range]
    agents = [eps_agents, ns_eps_agents, opt_agents, grad_agents, ucb_agents]

    n_runs = 500
    max_steps = 1000
    perfs = [[] for _ in agents] # perf per agent algorithm
    for i, (rg, algo_agents) in enumerate(zip(ranges, agents)):
        print("Agent {}:".format(algo_agents[0].name))
        for agent in algo_agents:
            print("\tParameter {}".format(agent.tell_specs()))
            avg_perf_run = many_runs(bandit, agent, n_runs, max_steps, plotting=False)
            perfs[i].append(avg_perf_run.mean()) # average over 1000 first steps (over 10 runs average)

    # Plotting
    ## Titles and labels
    plt.title("Parameter study of bandit algorithms")
    plt.xlabel("Parameter value")
    plt.ylabel("Avg reward over {} first steps, avg over {} runs".format(max_steps, n_runs))
    ## Perfs length adaptation
    perfs[0] = perfs[0] + [None]*4
    perfs[1] = [None]*2 + perfs[1] + [None]*3
    perfs[2] = [None]*5 + perfs[2]
    perfs[3] = [None]*2 + perfs[3]
    perfs[4] = [None]*3 + perfs[4]
    ## Actual plotting
    xaxis = np.arange(len(total_range))
    plt.plot(xaxis, perfs[0], color='C1', label='EpsGreedy (epsilon)')
    plt.plot(xaxis, perfs[1], color='C2', label='NSEpsGreedy (alpha)')
    plt.plot(xaxis, perfs[2], color='C3', label='Optimistic (Q0)')
    plt.plot(xaxis, perfs[3], color='C4', label='Gradient (alpha)')
    plt.plot(xaxis, perfs[4], color='C5', label='UCB (c)')
    plt.xticks(xaxis, ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4'])
    plt.legend()
    ## Saving
    plot_name = "bandit/plots/param_study.png"
    plt.savefig(plot_name)
    print("Saved plot as {}".format(plot_name))
