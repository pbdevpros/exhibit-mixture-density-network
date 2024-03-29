#! env python

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class BasicNN(tf.keras.Model):

    def __init__(self, input_size, output_size, units):
        super(BasicNN, self).__init__()
        self.internal_layers = [
            tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=tf.keras.activations.tanh),
        ]
        self.output_layer = tf.keras.layers.Dense(output_size, kernel_initializer=tf.keras.initializers.GlorotUniform()) # no activation

    def call(self, x):
        for i in range(len(self.internal_layers)):
            layer = self.internal_layers[i]
            x = layer(x)
        return self.output_layer(x)

class MixtureDensityModel(tf.keras.Model):

    def __init__(self, input_size, output_size):
        super(MixtureDensityModel, self).__init__()
        self.internal_layers = [
            tf.keras.layers.Dense(20, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=tf.keras.activations.tanh),
        ]
        # 9 outputs correspond to 3 parameters for the 3 Gaussian kernel functions
        self.output_layer = tf.keras.layers.Dense(9, kernel_initializer=tf.keras.initializers.GlorotUniform()) # no activation

    def call(self, x):
        for i in range(len(self.internal_layers)):
            layer = self.internal_layers[i]
            x = layer(x)
        return self.output_layer(x)

class MixtureDensityModelError(tf.keras.losses.Loss):

    def __init__(self, num_kernels, **kwargs):
        super(MixtureDensityModelError, self).__init__()
        self.num_kernels = num_kernels

    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        z_mu = y_pred[:, :self.num_kernels]
        z_alpha = y_pred[:, self.num_kernels:self.num_kernels*2]
        z_sigma = y_pred[:, self.num_kernels*2:]

        E = tf.reduce_sum(self.compute_error(y_true, z_alpha, z_mu, z_sigma), 0)
        return -1 * tf.math.log(E)

    ######################################
    ####### internal functionality #######
    ######################################
        
    def compute_error(self, t, z_alpha, z_mu, z_sigma):
        '''
            Calculates a_i(x) * phi_i(t|x), for a single kernel (i.e. i = {1,2, OR ... , N})
        '''
        alpha = self.alpha(z_alpha)
        phi = self.phi(t, z_mu, z_sigma)
        
        return tf.transpose(alpha) *  phi

    def phi(self, t, z_mu, z_sigma):
        z_sigma = self.sigma(z_sigma)
        sigma = tf.linalg.diag(z_sigma) ** 2
        mu = z_mu
        mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(sigma))
        print(mvn.prob(t))
        return mvn.prob(t)

    def alpha(self, x):
        return tf.nn.softmax(x)

    def sigma(self, x):
        return tf.exp(x)

    ######################################
    ################ end ################
    ######################################