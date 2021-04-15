from baselines.common import plot_util as pu
import matplotlib.pyplot as plt


def plot_results(title, xlabel, ylabel, filename, results):
    r = results[0]
    plt.plot(r.progress['epoch'], r.progress['train/success_rate'])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)


if __name__ == '__main__':
    seed = 0
    env_id = 'PandaPickAndPlace-v0'
    plot_filename = f'{env_id}_{seed}_train'
    log_path = f'logs/PandaPickAndPlace_{seed}'
    results = pu.load_results(log_path)
    title = env_id
    xlabel = 'epochs'
    ylabel = 'success rate'
    plot_results(title, xlabel, ylabel, plot_filename, results)
