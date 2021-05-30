from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

def plot_results(title, xlabel, ylabel, filename, results):
    r = results[0]
    plt.plot(r.progress['epoch'], r.progress['train/success_rate'])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()

def save_result_plots():
    env_id = 'PandaPickAndPlace-v0'
    for seed in range(3):
        plot_filename = f'{env_id}_{seed}_train.png'
        log_path = f'logs_PandaPickAndPlace_seed{seed}/'
        results = pu.load_results(log_path)
        title = env_id
        xlabel = 'epochs'
        ylabel = 'success rate'
        plot_results(title, xlabel, ylabel, plot_filename, results)

def generate_plot(metric='train'):
    env_id = 'PandaPickAndPlace-v0'
    title = f'{env_id}_{metric}'
    _, axes = plt.subplots()
    axes.set_title(title)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Success rate")
    log_paths = [f'logs_PandaPickAndPlace_seed{seed}/' for seed in range(3)]
    results = [pu.load_results(log_path)[0] for log_path in log_paths]
    xvalues = np.array(results[0].progress['epoch'])
    yvalues = np.array([r.progress[f'{metric}/success_rate'] for r in results])
    result_mean = np.mean(yvalues, axis=0)
    result_std = np.std(yvalues, axis=0)
    # Plot learning curve
    axes.grid()
    axes.fill_between(xvalues, result_mean - result_std,
                         result_mean + result_std, alpha=0.1,
                         color="r")
    axes.plot(xvalues, result_mean, '-', color="r", label='Panda (bullet)')
    axes.legend(loc="best")
    filename = f'{env_id}_seeds_{metric}.png'
    plt.savefig(filename)


if __name__ == '__main__':
    # save_result_plots()
    for metric in ['train', 'test']:
        generate_plot(metric)
