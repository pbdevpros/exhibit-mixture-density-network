#! env python

import tensorflow as tf
from utilities import plot_metric 

def generate_data_sinx():
    batch_size = 100000
    end = int(batch_size * 0.8)
    x = tf.random.uniform((batch_size,1), minval=-10, maxval=10)
    y = tf.math.sin(x)
    return x[:end], y[:end], x[end:], y[end:] # train_x, train_y, test_x, test_y

def generate_big_data_sinx():
    start = tf.ones((1000,1)) * -5
    stop = tf.ones((1000,1)) * 5
    x = tf.linspace(start, stop, 100, axis=1)
    x = tf.squeeze(x)
    y = tf.math.sin(x)
    return x[:500], y[:500], x[500:], y[500:] # train_x, train_y, test_x, test_y