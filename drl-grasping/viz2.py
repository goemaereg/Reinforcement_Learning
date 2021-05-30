from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

def plot_results(title, xlabel, ylabel, filename, results):
    r = results[0]
    epochs = list(range(len(r.progress)))
    okrate = r.progress['train/success_rate']
    plt.plot(epochs, okrate)
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

def get_results(env_id, metric, seeds):
    log_paths = [f'results/{env_id}/{seed}/' for seed in seeds]
    results = [pu.load_results(log_path)[0] for log_path in log_paths]
    if f'{metric}/success_rate' in results[0].progress.columns:
        xvalues = np.array([r.progress['epoch'] for r in results])
        yvalues = np.array([r.progress[f'{metric}/success_rate'] for r in results])
    else:
        xvalues = np.array([r.progress['total/epochs'] for r in results])
        ret = np.array([r.progress['rollout/return'] for r in results])
        tot = np.array([r.progress['rollout/episode_steps'] for r in results])
        yvalues = (tot + ret) / tot
    result_mean = np.mean(yvalues, axis=0)
    result_std = np.std(yvalues, axis=0)
    return xvalues, yvalues, result_mean, result_std

def generate_plot(envs, metric='train', seeds=range(3)):
    title = f'PandaPickAndPlace-{metric}'
    _, axes = plt.subplots()
    axes.set_title(title)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Success rate")
    results = [get_results(env_id, metric, seeds) for env_id in envs]
    # Plot learning curve
    axes.grid()
    colors=['r', 'b']
    for i, _ in enumerate(envs):
        xvalues, yvalues, ymean, ystd = results[i]
        xvalues = xvalues[i][:50]
        ymean = ymean[:50]
        ystd = ystd[:50]
        axes.fill_between(xvalues, ymean - ystd,
                             ymean + ystd, alpha=0.1,
                             color=colors[i])
        axes.plot(xvalues, ymean, '-', color=colors[i], label=envs[i])
    axes.legend(loc="best")
    filename = f'PandaPickAndPlace_seeds_{metric}.png'
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':
    # save_result_plots()
    # envs = ['PandaPickAndPlace-v0', 'PandaPickAndPlace-v1']
    seeds = range(1)
    envs = ['PandaPickAndPlace-v1']
    #seeds = range(3)
    for metric in ['train', 'test']:
        generate_plot(envs, metric, seeds=seeds)
