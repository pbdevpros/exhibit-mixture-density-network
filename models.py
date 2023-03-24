#! env python

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class BasicNN(tf.keras.Model):

    def __init__(self, input_size, output_size):
        super(BasicNN, self).__init__()
        self.internal_layers = [
            tf.keras.layers.Dense(64, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=tf.keras.activations.softmax),
            tf.keras.layers.Dense(64, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=tf.keras.activations.softmax)
        ]
        self.output_layer = tf.keras.layers.Dense(output_size, kernel_initializer=tf.keras.initializers.GlorotUniform()) # no activation

    def call(self, x):
        for i in range(len(self.internal_layers)):
            layer = self.internal_layers[i]
            x = layer(x)
        return self.output_layer(x)

class RandomGeneratorNN(tf.keras.Model):

    def __init__(self, input_size, output_size, units):
        super(RandomGeneratorNN, self).__init__()
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
        # mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        return mvn.prob(t)

    # def phi(self, t, z_mu, z_sigma):
    #     z_sigma = self.sigma(z_sigma ** 2) 
    #     # sigma = tf.linalg.diag(z_sigma) ** 2
    #     mu = z_mu
    #     v = []
    #     for i in range(self.num_kernels):
    #         mvn = tfd.MultivariateNormalTriL(loc=mu[:,i], scale_tril=tf.linalg.cholesky(z_sigma[:,i]))
    #         print(mvn.prob(t))
    #         # mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
    #         v.append(mvn.prob(t))
    #     return tf.Variable(v)

    def alpha(self, x):
        return tf.nn.softmax(x)

    def sigma(self, x):
        return tf.exp(x)

    ######################################
    ################ end ################
    ######################################


class MixtureDensityModelErrorFinal(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super(MixtureDensityModelErrorFinal, self).__init__()
        self.z_alpha = None
        self.z_mu = None
        self.z_sigma = None

    # def call(self, y_true, y_pred, sample_weight=None):
    #     y_pred = tf.cast(y_pred, dtype=tf.float32)
    #     self.z_mu = y_pred[:, :2]
    #     self.z_sigma = y_pred[:, 2:]
    #     B, N = self.z_sigma.shape
    #     self.z_sigma = tf.reshape(self.z_sigma, (B, int(N/2), int(N/2)))
    #     print(self.z_sigma.shape)
    #     covariance = self.z_sigma @ tf.transpose(self.z_sigma, perm=[0, 2, 1])
    #     print(self.z_mu.shape)
    #     print(covariance.shape)
    #     v = []
    #     for i in range(B):
    #         mvn = tfd.MultivariateNormalTriL(loc=self.z_mu[i], scale_tril=covariance[i])
    #         v.append(mvn.prob(y_true[i]))
    #     print(len(v))
    #     v = tf.Variable(v)
    #     E = tf.reduce_sum(v, 0)
    #     return -1 * tf.math.log(E)

    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        self.z_mu = y_pred[:, :2]
        self.z_sigma = y_pred[:, 2:]
        B, N = self.z_sigma.shape
        self.z_sigma = tf.reshape(self.z_sigma, (B, int(N/2), int(N/2)))
        print(self.z_sigma.shape)
        covariance = self.z_sigma @ tf.transpose(self.z_sigma, perm=[0, 2, 1])
        print(self.z_mu.shape)
        print(covariance.shape)
        
        def func_map(x):
            mvn = tfd.MultivariateNormalTriL(loc=self.z_mu[x], scale_tril=covariance[x])
            mvn.prob(y_true[x])
        v = list(map(func_map, range(B)))
        # v = []
        # for i in range(B):
        #     mvn = tfd.MultivariateNormalTriL(loc=self.z_mu[i], scale_tril=covariance[i])
        #     v.append(mvn.prob(y_true[i]))
        # print(len(v))
        v = tf.Variable(v)
        E = tf.reduce_sum(v, 0)
        return -1 * tf.math.log(E)


# def mdn_loss(y_est, y):
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     return loss_object(y_true=y, y_pred=y_est)
#     # y = tf.cast(y, dtype=tf.float32)
#     # return tf.math.reduce_euclidean_norm(y_est - y)

def gaussian_2d(mean, sigma):
    import numpy as np
    np.random.multivariate_normal(mean, cov, 10000)


# loss = MixtureDensityModelErrorFinal(num_kernels=3)
# y_true = tf.random.uniform((32, 1), minval=-10, maxval=10)
# # y_pred = tf.Variable([[3, 3, 3, 4, 4, 4, 1, 1, 1]])
# y_pred = tf.Variable([[3, 3, 0.8, 0, 1.2, 1.1]])
# y_pred = tf.tile(y_pred, tf.constant([32, 1]))
# # # y_pred = tf.repeat(y_pred, [32, 1])
# print(loss(y_true, y_pred))

y_pred = tf.Variable([[0.8, 1.2, 1.1]])
y_pred = tf.tile(y_pred, tf.constant([32, 1]))
print(y_pred)
L = tfp.math.fill_triangular(y_pred)
sigma = tf.matmul(L, tf.transpose(L, perm=[0, 2, 1]))
print(sigma)

y_pred = tf.Variable([0.8, 1.2, 1.1])
L = tfp.math.fill_triangular(y_pred)
sigma = tf.matmul(L, tf.transpose(L))
print(sigma)