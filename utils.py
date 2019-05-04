import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def assert_not_abstract(obj, abstract_name):
    """ Makes sure the obj isn't a instance of the abstract class calling this.
        """
    assert obj.__class__.__name__ != abstract_name, \
           "Cannot instantiate class " + abstract_name + \
           " as it is assumed abstract."

def save_plot(l, file_name, title, xlabel, ylabel, xaxis=None, force_xaxis_0=False):
    """ Simply saves a plot with multiple usual arguments.
        """
    if xaxis is None:
        xaxis = np.arange(len(l))
    plt.plot(xaxis,l)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
