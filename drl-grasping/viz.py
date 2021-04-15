from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results):
    r = results[0]
    plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))


if __name__ == '__main__':
    seed = 0
    log_path = '~/logs/PandaPickAndPlace_{}/'.format(seed)
    env_id = 'PandaPickAndPlace-v0'
    results = pu.load_results('~/logs/cartpole-ppo')
