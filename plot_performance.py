import matplotlib.pyplot as plt
import numpy as np


results2 = [
    {
        'env' : 'FourRoomsGoal-v0',
        'title': 'Q-Learning performance',
        'subtitle': 'eps=0.1, lr=0.05, gamma=0.9',
        'labelx': 'optimization steps',
        'labely': 'episode performance (steps to goal)',
        'data': {
            'count': {
                'label': 'HERModel',
                'array': 'outputm/train_stl_28_HER_FourRoomsGoal-v0_QLearning_perf_3.plot.npy',
                },
            'her': {
                'label': 'tabular_her',
                'array': 'output2/her_tabular_FourRoomsGoal-v0_QLearning_perf_subtraject_len_28_3.npy'
            }
        }
    }
    ]
results = [
    {
        'env' : 'FourRoomsGoal-v0',
        'title': 'Q-Learning performance',
        'subtitle': 'eps=0.1, lr=0.05, gamma=0.9',
        'labelx': 'optimization steps',
        'labely': 'episode performance (steps to goal)',
        'data': {
            'count': {
                'label': 'tabular',
                'array': 'output/count_tabular_FourRoomsGoal-v0_QLearning_perf_3.npy',
                },
            'her': {
                'label': 'tabular_her',
                'array': 'output/her_tabular_FourRoomsGoal-v0_QLearning_perf_n_subtraject_steps_60_3.npy'
            }
        }
    },
    {
        'env' : 'FourRoomsGoalBig-v0',
        'title': 'Q-Learning performance',
        'subtitle': 'eps=0.1, lr=0.05, gamma=0.9',
        'labelx': 'optimization steps',
        'labely': 'episode performance (steps to goal)',
        'data': {
            'count': {
                'label': 'tabular',
                'array': 'output/count_tabular_FourRoomsGoalBig-v0_QLearning_perf_10.npy',
                },
            'her': {
                'label': 'tabular_her',
                'array': 'output/her_tabular_FourRoomsGoalBig-v0_QLearning_perf_n_subtraject_steps_44_10.npy'
            }
        }
    }
]

def save_plot(data, file_name, title, subtitle, xlabel, ylabel):
    # fig, ax = plt.subplots(1, 1)
    plt.suptitle(subtitle, fontsize=14, fontweight='bold')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for k, v in data.items():
        label = v['label']
        arr = np.load(v['array'])
        xaxis = arr[0]
        yaxis = arr[1]
        # plt.xticks(xaxis, xaxis)
        # ax.plot(xaxis, yaxis, marker='o', label=label)
        # smoothing
        n_episodes = xaxis.size
        smooth_avg = n_episodes // 100
        smooth = np.zeros(n_episodes)
        l = np.array(yaxis)
        l_smooth = [None for _ in range(smooth_avg)]
        l_smooth += [ np.mean(l[i - smooth_avg:i + smooth_avg])
                     for i in range(smooth_avg, len(l) - smooth_avg) ]
        indices = slice(0, len(l) - smooth_avg, 1)
        # i=30
        # print(f'{label} x[0] = {xaxis[i]}, y[0] = {yaxis[i]}, ys = {l_smooth[i]}')

        plt.plot(xaxis[indices], l_smooth[indices], label=label)
        plt.axis((0, 48000, 0, 150))
        # plt.plot(xaxis, yaxis, label=label)
    file_name += '.png'
    plt.legend()
    plt.savefig(file_name)
    print("Saved figure to", file_name)
    plt.close()


def main():
    for d in results2:
        filename = f"output2/perf_{d['env']}"
        title = d['title']
        subtitle = d['subtitle']
        xlabel = d['labelx']
        ylabel = d['labely']
        save_plot(d['data'], filename, title, subtitle, xlabel, ylabel)


if __name__ == '__main__':
    main()
