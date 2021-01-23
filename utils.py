import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from time import time
import random
import os
import gym

def classname(obj):
    return obj.__class__.__name__

def envname(env):
    """ Name of a gym environment, without the version"""
    return env.unwrapped.spec.id[:-3]

def assert_not_abstract(obj, abstract_name):
    """ Makes sure the obj isn't a instance of the abstract class calling this.
        """
    assert obj.__class__.__name__ != abstract_name, \
           "Cannot instantiate class " + abstract_name + \
           " as it is assumed abstract."

def save_plot(l, file_name, suptitle, title, xlabel, ylabel,
              xaxis=None, force_xaxis_0=False, interval_xaxis=None, interval_yaxis=None,
              smooth_avg=0, only_avg=False, labels=None, xlineat=None,
              ylineat=None):
    """ Simply saves a plot with multiple usual arguments.
        smooth_avg > 0 adds a smoothed curve over 2*(the number of surrounding
            episodes given)(moving averages)
        """
    plt.figure()
    if xaxis is None:
        if (l.ndim == 1):
            xaxis = np.arange(len(l))
        else:
            xaxis = np.arange(l.shape[-1])
    if not only_avg:
        if labels is None:
            plt.plot(xaxis,l)
        else:
            for perf, label in zip(l, labels):
                plt.plot(xaxis, perf, label=label)
            plt.legend()
    else:
        plt.plot(xaxis,len(l)*[None])
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if smooth_avg > 0:
        # not the most efficient but most intuitive
        l = np.array(l)
        l_smooth = [None for _ in range(smooth_avg)]
        l_smooth += [np.mean(l[i-smooth_avg:i+smooth_avg])
                     for i in range(smooth_avg, len(l)-smooth_avg)]
        plt.plot(xaxis[: len(l)-smooth_avg], l_smooth)

    if force_xaxis_0:
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,y2))

    if interval_xaxis is not None:
        new_x1, new_x2 = interval_xaxis
        x1,x2,y1,y2 = plt.axis()
        plt.axis((new_x1,new_x2,y1,y2))

    if interval_yaxis is not None:
        new_y1, new_y2 = interval_yaxis
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,new_y1,new_y2))

    if xlineat is not None:
        plt.axhline(xlineat, color='red', linewidth=0.5)
    if ylineat is not None:
        plt.axvline(ylineat, color='red', linewidth=0.5)

    file_name += '.png'
    ensure_dir(file_name)
    plt.savefig(file_name)
    print("Saved figure to", file_name)
    #plt.close()

def smooth(perf, smooth_avg):
    perf_smooth = []
    perf_smooth += [np.mean(perf[i-smooth_avg:i+smooth_avg])
                    for i in range(smooth_avg, len(perf)-smooth_avg)]
    return perf_smooth

def str2class(class_string):
    try:
        return globals()[class_string]
    except KeyError as ke:
        print(ke, "Tried to access an undefined class :", class_string)

def my_timeit(f, **kwargs):
    """ Returns the function execution time"""
    init = time()
    f(**kwargs)
    return time() - init

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def my_random_choice(v, p=None):
    """ Faster version of the np.random.choice function with probabilities """
    if p is None:
        return v[np.random.randint(len(v))]
    # else (general case)
    assert (abs(sum(p)-1.)<1e-6), "Invalid probability vector p, sum={}".format(sum(p))
    r = np.random.rand()
    i = 0
    s = p[i]
    while s < r:
        i += 1
        s += p[i]

    if type(v) is int:
        assert len(p) == v, "Int doesn't match proba length: {} != {}".format(v, len(p))
        return i
    else:
        assert len(v) == len(p), "Incorrect entry lengths v,p: {} != {}".format(len(v), len(p))
        return v[i]

def allmax(a):
    """ Returns all occurences of the max """
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_

def my_argmax(v):
    """ Breaks ties randomly. """
    return random.choice(allmax(v))

def prod(v):
    p = 1
    for e in v:
        p *= e
    return p

def softmax(x, T=1):
    """Computes softmax values for each element of vector x,
    i.e. softmax(x_i) = e^x_i / sum_j (e^x_j).
    Substracting the max doesn't change the output but adds numerical stability.
    T is the temperature and controls the stochasticity (high=lots low=nots)
    """
    x /= T
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

class TileCoder:
    """ Copied from https://github.com/MeepMoop/tilecoding/blob/master/tilecoding.py
    Creates a tiling of an environment."""
    def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
        tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int) + 1
        self._offsets = offset(len(tiles_per_dim)) * \
                        np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1
        self._limits = np.array(value_limits)
        self._norm_dims = np.array(tiles_per_dim) / (self._limits[:, 1] - self._limits[:, 0])
        self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
        self._hash_vec = np.array([np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))])
        self._n_tiles = tilings * np.prod(tiling_dims)

    def __getitem__(self, x):
        off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
        return self._tile_base_ind + np.dot(off_coords, self._hash_vec)

    @property
    def n_tiles(self):
        return self._n_tiles


def one_hot(v, n_elements):
    z = np.zeros(n_elements)
    z[v] = 1
    return z


import pylab
def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = pylab.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None:
        label_prop = dict()
    pylab.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)
