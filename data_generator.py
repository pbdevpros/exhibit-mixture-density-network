#! env python

import math
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

def sinusoid_random(x):
    x = tf.cast(x, dtype=tf.float32)
    return x + 0.3 * tf.math.sin(2*tf.constant(math.pi*x)) +  tf.random.uniform(x.shape, minval=-0.1, maxval=0.1)

def generate_data_sinusoid():
    batch_size = 1250 # 1250 * 0.8 = 1000 samples for training
    end = int(batch_size * 0.8)
    x = tf.random.uniform((batch_size,1), minval=0, maxval=1)
    y = sinusoid_random(x)
    return x[:end], y[:end], x[end:], y[end:] # train_x, train_y, test_x, test_y