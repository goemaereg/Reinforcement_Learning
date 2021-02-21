import matplotlib.pyplot as plt


### FourRoomsGoal-v0
envs = {
    'FourRoomsGoal-v0': {
        4: 4.0,
        12: 4.0, 
        20: 3.0,  
        28: 2.3,
        36: 3.7,  
        44: 4.7,  
        52: 4.7,  
        60: 4.3  
    },
    'FourRoomsGoalBig-v0': {
        4: 392.2,
        12: 359.2,  
        20: 260.9,  
        28: 220.2,  
        36: 275.9,  
        44: 119.5,
        52: 304.1,  
        60: 161.4  
    }
}


def save_plot(xaxis, yaxis, file_name, title, subtitle, xlabel, ylabel):
    fig, ax = plt.subplots(1, 1)
    plt.suptitle(subtitle, fontsize=14, fontweight='bold')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xaxis, xaxis)
    ax.plot(xaxis, yaxis, marker='o')
    file_name += '.png'
    fig.savefig(file_name)
    print("Saved figure to", file_name)
    plt.close()


def main():
    for e,d in envs.items():
        print(e)
        k = list(d.keys())
        v = list(d.values())
        filename = f'subtraject_{e}'
        title = 'subtraject hyperparameter optimization'
        subtitle = e
        xlabel = 'subtraject length'
        ylabel = 'final performance'
        save_plot(k, v, filename, title, subtitle, xlabel, ylabel)


if __name__ == '__main__':
    main()
