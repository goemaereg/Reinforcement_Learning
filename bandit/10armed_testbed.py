import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def epsilon_greedy(epsilon, Q):
    """ Simple epsilon-greedy strategy given action-value estimator Q
        """
    return np.random.randint(len(Q)) if np.random.rand() < epsilon \
                                     else np.argmax(Q)

def compute_reward(q_a):
    return np.random.randn() + q_a

def run(epsilon, n_actions=10, max_steps=1000):
    """ Runs a single (epsilon-greedy / sample-average) strategy on a
        n_actions-armed bandit problem.
        """
    q = np.random.randn(n_actions)  # true action-value functions
    Q = np.zeros(n_actions)         # q estimator
    visits = np.zeros(n_actions)    # number of visits of each action
    reward_record = np.empty(max_steps)
    norm_record = np.empty(max_steps)
    for step in range(max_steps):
        action = epsilon_greedy(epsilon, Q)
        reward = compute_reward(q[action])
        visits[action] += 1
        Q[action] += (reward - Q[action])/visits[action] # mean increment

        norm = np.linalg.norm(Q-q)
        norm_record[step] = norm
        reward_record[step] = reward

    return reward_record, norm_record

def save_plot(l, file_name, title, xlabel, ylabel, xaxis=None, force_xaxis_0=False):
    """ Simply saves a plot with multiple usual arguments.
        """
    if xaxis is None:
        xaxis = np.arange(len(l))
    plt.plot(xaxis,l)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if force_xaxis_0:
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,y2))
    file_name = './plots/'+file_name + '.png'
    plt.savefig(file_name)
    print("Saved figure to", file_name)
    plt.close()

def many_runs(n_runs, epsilon, max_steps=2000, plotting=True):
    """ Runs n_runs of the bandit problem with an epsilon_greedy.
        We record and return the rewards and norms
        We plot their average over the runs if 'plotting'.
        """
    reward_record_runs = np.empty((n_runs, max_steps))
    norm_record_runs = np.empty((n_runs, max_steps))
    for r in range(n_runs):
        reward_record_runs[r], norm_record_runs[r] = run(epsilon, \
                                                         max_steps=max_steps)

    perf = np.mean(reward_record_runs, axis=0)
    norm = np.mean(norm_record_runs, axis=0)
    if plotting:
        save_plot(perf, 'perf'+str(epsilon), 'Average over '+str(n_runs)+' runs : Reward over time for eps='+str(epsilon), xlabel='Step', ylabel='Reward')
        save_plot(norm, 'norm'+str(epsilon), 'Average over '+str(n_runs)+' runs : Norm of (Q-q) for eps='+str(epsilon), xlabel='Step', ylabel='Norm (Q-q)')

    return perf, norm

def eps_spectrum(n_epsilons, n_runs, n_last_perf=10, plotting=True):
    """ Compares performance and converge of a whole spectrum of epsilons.
        n_last_perf gives the number of steps to consider for final averaging.
        """
    comp_perf = np.empty(n_epsilons)
    comp_norm = np.empty(n_epsilons)
    epsilons = np.linspace(0,.2,n_epsilons)
    for i,epsilon in enumerate(epsilons):
        perf, norm = many_runs(n_runs, epsilon, plotting=False)
        comp_perf[i] = np.mean(perf[-n_last_perf:])
        comp_norm[i] = np.mean(norm[-n_last_perf:])
        print("[", comp_perf[i], ",", epsilon, "],")

    if plotting:
        save_plot(comp_perf, 'perf_spectrum', 'Final performances over '+str(n_runs)+' runs : reward comparison for varying epsilon', xlabel='Epsilon', ylabel='Average rewards over last '+str(n_last_perf)+' steps', xaxis=epsilons)
        save_plot(comp_norm, 'norm_spectrum', 'Final norms over '+str(n_runs)+' runs : norm(Q-q) comparison for varying epsilon', xlabel='Epsilon', ylabel='Average norm over last '+str(n_last_perf)+' steps', xaxis=epsilons)

    return comp_norm, comp_perf


if __name__ == '__main__':
    n_runs = 1000
    n_epsilons = 21
    eps_spectrum(n_epsilons, n_runs)
