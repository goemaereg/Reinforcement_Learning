import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def assert_not_abstract(obj, abstract_name):
    """ Makes sure the obj isn't a instance of the abstract class calling this.
        """
    assert obj.__class__.__name__ != abstract_name, \
           "Cannot instantiate class " + abstract_name + \
           " as it is assumed abstract."

def save_plot(l, file_name, title, xlabel, ylabel,
              xaxis=None, force_xaxis_0=False, smooth_avg=0, only_avg=False,
              labels=None):
    """ Simply saves a plot with multiple usual arguments.
        smooth_avg > 0 adds a smoothed curved over 2*the number of surrounding episodes given
        """
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
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if smooth_avg > 0:
        # not the most efficient but most intuitive
        l = np.array(l)
        l_smooth = [None for _ in range(smooth_avg)]
        l_smooth += [np.mean(l[i-smooth_avg:i+smooth_avg])
                     for i in range(smooth_avg, len(l)-smooth_avg)]
        plt.plot(l_smooth)

    if force_xaxis_0:
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,y2))

    file_name += '.png'
    plt.savefig(file_name)
    print("Saved figure to", file_name)
    plt.close()

def str2class(class_string):
    try:
        return globals()[class_string]
    except KeyError as ke:
        print(ke, "Tried to access an undefined class :", class_string)

def my_random_choice(v, p=None):
    """ Faster version of the np.random.choice function with probabilities """
    if p is None:
        return v[np.random.randint(len(v))]
    # else (general case)
    assert (abs(sum(p)-1.)<1e-6); "Invalid probability vector p"
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
