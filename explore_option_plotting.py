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
    # 'tabular/np_arrays/ExploreOption2/Inverse_sqrt/p_ex.npy',
    # 'tabular/np_arrays/ExploreOption2/Negative_sqrt/p_ex.npy',
    # 'tabular/np_arrays/ExploreOption2/Successor_Rep/p_ex.npy',
    # 'tabular/np_arrays/ExploreOption_Multi/p_ex.npy',
    # 'tabular/np_arrays/ExploreOption2/Random_Reward/random_explorer.npy',
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
spectrum = [0, 0.001, 0.005, 0.01, 0.05, 0.075, 0.1]
# spectrum = [0,0.2,0.35,0.5,0.75,1]
# spectrum = [0,0.001, 0.005,0.01,0.05,0.1,0.2]
perfs = [np.load(file_name)[1:] for file_name in file_names]
print(perfs[0])
xys = [perf_to_avg(perf,300) for perf in perfs]

# launch_specs = 'QL+EO_intrinsic_rewards'
launch_specs = 'QL+VC_beta_dotted'
file_name = "tabular/perf_plots/custom/{}/{}".format(env_name, launch_specs)
# suptitle = "QL+EO Intrinsic Reward Functions"
suptitle = "QL+VC varying beta"
title = 'lr=0.1, lrEO=0.001, eps=0.1, g=0.9, p_ex=0.5'
# title = 'lr=0.1, lrEO=0.001, eps=0.1, g=0.9, c_switch=15'
# title = 'lr=0.1, eps=0.1, g=0.9, c_switch=15, p_ex=0.5'
# title = 'lr=0.1, eps=0.1, g=0.9'
#xlabel = '100 Time steps'
#ylabel = "Steps to goal"
xlabel = 'First phase'
ylabel = 'Second phase'

## PLOTTING --------------------------------------------------------------------

ax0 = plt.subplot(211)
ax1 = plt.subplot(212, sharex=ax0)
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=0.1)

for i,(x,y) in enumerate(xys):
    n_points = len(spectrum)
    if i==0:
        ax0.plot     (range(n_points), x, linestyle=':', label=labels[i])
    else:
        ax0.plot     (range(n_points), x, label=labels[i])
    ax0.scatter  (range(n_points), x, s=5)
    if i==0:
        ax1.plot     (range(n_points), y, linestyle=':', label=labels[i])
    else:
        ax1.plot     (range(n_points), y, label=labels[i])
    ax1.scatter  (range(n_points), y, s=5)
# plt.legend()
ax0.legend()

plt.suptitle(suptitle, fontsize=14, fontweight='bold')
ax0.set_title(title)
ax0.set_ylabel(xlabel, labelpad=-5)
ax1.set_ylabel(ylabel, labelpad=1)
suplabel('x', r'$\beta$', labelpad=7)
# suplabel('x', r'$\alpha_{EO}$', labelpad=7)
# suplabel('x', r'$c_{switch}$', labelpad=7)
# suplabel('x', r'$\beta$', labelpad=7)
suplabel('y', 'Average time to the goal during:', labelpad=10)

ax0.set_ylim(16, 100)
ax0.set_yticks([16, 40,60,80,100])
ax1.set_ylim(10, 16)
plt.xticks(range(len(spectrum)),np.array(spectrum,dtype=str))

# QLearning performance in first phase
ax0.axhline(35.,color='red',ls='--', lw=1)
ax0.text(len(spectrum)-.5,33, "QL", fontsize=10,color='red')

#fig.axis((16,100,10,16))

## FINAL SAVING AND STUFF ------------------------------------------------------
file_name += '.png'
ensure_dir(file_name)
plt.savefig(file_name)
print("Saved figure to", file_name)
plt.close()
