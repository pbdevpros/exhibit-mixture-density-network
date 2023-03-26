#! env python

import numpy as np
import pickle

def save_history(history, path):
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)

def load_history(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_histories(histories):
    x = []
    y = []
    for history in histories:
        length = len(history.history['loss'])
        y.append(history.history['loss'])
        #  = history.history['accuracy']
        x.append(np.linspace(0, length, length))
        
    plot_metric(x, y, 'Training loss of CNNs using different loss functions', [])

def gaussian_2d(mean, sigma):
    import numpy as np
    np.random.multivariate_normal(mean, sigma, 10000)

def plot_metric(x, y, title, legends):
    import matplotlib.pyplot as plt
    if type(x) != type([]): x = [x]
    if type(y) != type([]): y = [y]
    assert(len(x) == len(y))
    colors = [ 'g*', 'r*', 'b*']
    for i in range(len(x)):
        plt.plot(x[i], y[i], colors[i % len(colors)])
    plt.legend(legends, loc='upper right')
    plt.title(title)
    plt.rcParams["figure.figsize"] = (30, 30)
    plt.grid()
    plt.rc({'font.size': 42})
    plt.show()

def plot_xy(x, y, title, legend):
    import matplotlib.pyplot as plt
    assert(len(x) == len(y))
    plt.plot(x, y, 'r*')
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.rcParams["figure.figsize"] = (30, 30)
    plt.grid()
    plt.rc({'font.size': 42})
    plt.show()