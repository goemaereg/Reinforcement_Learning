from utils import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

## FUNCTIONS -------------------------------------------------------------------
def perf_to_avg(perf, N):
    """ Turns a perf list (1000 entries) into the average convergence time
    from - beginning to N (path close); - N to end (path opened)
    We suppose the perf is in usual form, i.e. the 2nd dim is the steps,
    hence we flip it here."""
    perf = perf.T
    before = np.clip(sum(perf[:N])/N, 16,100)
    after = np.clip(sum(perf[N:])/(len(perf)-N), 10,16)
    return before, after


## ENV STUFF -------------------------------------------------------------------
env_name = 'ShortcutMaze'
## LOADING ---------------------------------------------------------------------
# Warning! First perf is optimal here.
file_names = [
    'tabular/np_arrays/ExploreOption2/Inverse_sqrt/p_ex.npy',
    'tabular/np_arrays/ExploreOption2/Negative_sqrt/p_ex.npy',
    'tabular/np_arrays/ExploreOption2/Successor_Rep/p_ex.npy',
    'tabular/np_arrays/ExploreOption_Multi/c_switches.npy',
    'tabular/np_arrays/ExploreOption2/Random_Reward/random_explorer.npy',
]
file_names = [
    'tabular/np_arrays/QLearning_VC/Inverse_sqrt/betas.npy',
    'tabular/np_arrays/QLearning_VC/Negative_sqrt/betas.npy',
    'tabular/np_arrays/QLearning_VC/Successor_Rep/betas.npy'
]

labels = [
   'Inverse_sqrt',
   'Negative_sqrt',
   'Successor_Rep',
    # 'All',
    # 'Random_Explorer',
]
# spectrum = [1,3,5,7,10,15,20,30,60,3600]
# spectrum = [0, 0.001, 0.005, 0.01, 0.05, 0.075, 0.1]
# spectrum = [0,0.2,0.35,0.5,0.75,1]
spectrum = [0,0.001, 0.005,0.01,0.05,0.1,0.2]
perfs = [np.load(file_name)[1:] for file_name in file_names]
print(perfs[0])
xys = [perf_to_avg(perf,300) for perf in perfs]

# launch_specs = 'QL+EO_intrinsic_rewards'
launch_specs = 'QL+EO_p_ex_tt'
file_name = "tabular/perf_plots/custom/{}/{}".format(env_name, launch_specs)
# suptitle = "QL+EO Intrinsic Reward Functions"
suptitle = "QL+EO varying p_ex"
# title = 'lr=0.1, lrEO=0.001, eps=0.1, g=0.9, p_ex=0.5'
title = 'lr=0.1, lrEO=0.001, eps=0.1, g=0.9, c_switch=15'
# title = 'lr=0.1, eps=0.1, g=0.9, c_switch=15, p_ex=0.5'
# title = 'lr=0.1, eps=0.1, g=0.9'
#xlabel = '100 Time steps'
#ylabel = "Steps to goal"
xlabel = 'First phase'
ylabel = 'Second phase'

## PLOTTING --------------------------------------------------------------------
for i,(x,y) in enumerate(xys):
    for j, txt in enumerate(spectrums[i]):
        plt.annotate(txt, (x[j], y[j]))
    plt.plot(x, y, label=labels[i])
    plt.scatter(x, y)
    print(i)
plt.legend()


plt.suptitle(suptitle, fontsize=14, fontweight='bold')
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

## FINAL SAVING AND STUFF ------------------------------------------------------
file_name += '.png'
ensure_dir(file_name)
plt.savefig(file_name)
print("Saved figure to", file_name)
plt.close()
