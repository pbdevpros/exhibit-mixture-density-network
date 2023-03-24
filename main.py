#! env python
# Fundamental experiments with Mixed Density Networks

import tensorflow as tf
import models as ml
import numpy as np
import data_generator as dg
from data_generator import generate_data_sinx
from utilities import save_history, load_history, plot_histories, plot_xy, plot_metric

import tensorflow_probability as tfp
tfd = tfp.distributions

####################################################################################
####################################################################################
############################### Helper functions ###################################
####################################################################################

def compile_and_train(model, loss, x_train, y_train, epochs):
    model.compile(
              optimizer='adam',
              loss=loss,
              metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs)
    return history

####################################################################################
####################################################################################
############################### BasicNN  ###########################################
####################################################################################

def train_and_predict_basic_nn(epochs):
    # dataset
    (x_train, y_train, x_test, y_test) = generate_data_sinx()

    # define model 
    model = ml.BasicNN(x_train.shape[1], y_train.shape[1])
    loss = tf.keras.losses.MeanSquaredError()

    # training
    history = compile_and_train(model, loss, x_train, y_train, epochs)
    print("Evaluating model: " + model.name)

    # evaluation
    (l, acc) = model.evaluate(x_test, y_test, verbose=2)
    print("Achieved loss: {}, accuracy: {} ".format(l, acc))

    # prediction
    x = tf.linspace(-5, 5, 100)
    y = model.predict(tf.transpose([x]))
    y1 = tf.math.sin(x)
    plot_metric([x, x], [y, y1], "Plot of neural net approximating sin (x)", ["nn(x)", "sin(x)"])

def toy_problem_fx(dataset, epochs, units):
    # dataset
    (x_train, y_train, x_test, y_test) = dataset

    # define model
    model = ml.RandomGeneratorNN(x_train.shape[1], y_train.shape[1], units)
    loss = tf.keras.losses.MeanSquaredError()

    # training
    history = compile_and_train(model, loss, x_train, y_train, epochs)
    print("Evaluating model: " + model.name)

    # evaluation
    (l, acc) = model.evaluate(x_test, y_test, verbose=2)
    print("Achieved loss: {}, accuracy: {} ".format(l, acc))

    return model

def train_and_predict_forward_pass(epochs):
    dataset = dg.generate_data_sinusoid()
    x = tf.linspace(0, 1, 100)
    y1 = dg.sinusoid_random(x)
    toy_problem_fx(dataset, epochs, (x, y1), 5)

def train_and_predict_backward_pass(epochs):
    (y_train, x_train, y_test, x_test) = dg.generate_data_sinusoid()
    dataset = (x_train, y_train, x_test, y_test)
    model = toy_problem_fx(dataset, epochs, 20)
    
    # prediction
    t = tf.linspace(0, 1, 100)
    x_actual = dg.sinusoid_random(t)
    x_model = model.predict(tf.transpose([x_actual]))
    plot_metric([x_model, x_actual], [t, t], "Plot of neural net approximating f(x)", ["nn(x)", "f(x)"])

def train_and_predict_MDN(epochs):
    # dataset
    (y_train, x_train, y_test, x_test) = dg.generate_data_sinusoid()

    # define model
    model = ml.MixtureDensityModel(x_train.shape[1], y_train.shape[1])
    loss = ml.MixtureDensityModelError(num_kernels=3)

    # training
    history = compile_and_train(model, loss, x_train, y_train, epochs)
    print("Evaluating model: " + model.name)

    # evaluation
    (l, acc) = model.evaluate(x_test, y_test, verbose=2)
    print("Achieved loss: {}, accuracy: {} ".format(l, acc))

    # prediction
    t = tf.linspace(0, 1, 100)
    x_actual = dg.sinusoid_random(t)
    x_model = model.predict(tf.transpose([x_actual]))
    # o = y[0]
    # mu_s = o[0:3]
    # alpha_s = o[3:6]
    # sigma_s = o[6:]
    d = mdn_max_prediction(x_model)
    alphas = tf.nn.softmax(x_model[:,3:6])
    plot_metric([t, t, t ], [ alphas[:, 0], alphas[:, 1], alphas[:, 2]], "Plot of priors a_i(x)", ["a_1(x)", "a_2(x)", "a_3(x)"])
    plot_gaussian(t, x_model[0, 0], x_model[0, 6])
    
    # d = mdn_run_prediction(x, y)
    plot_metric([d, x_actual], [t, t], "Plot of neural net approximating f(x)", ["nn(x)", "f(x)"])

def mdn_max_prediction(y):
    # Eq. 49 in MDN, find the index (i) of the maximum alpha
    # Choose the i_th mean as a prediction of network
    alphas = y[:, 3:6]
    args = tf.math.argmax(alphas, axis=1)
    retval = []
    for i in range(y.shape[0]):
        retval.append(y[i, args[i]])
    return tf.Variable(retval)

def mdn_run_prediction(x, y):
    retval = []
    for i in range(len(x)):
        mu_s = y[i, 0:3]
        sigma_s = y[i, 6:]
        v = mdn_prediction_single(x[i], mu_s, sigma_s)
        retval.append(v)
    return tf.Variable(retval)

def mdn_prediction_single(x_i, mu_i, sigma_i):
    x_i = tf.cast(x_i, tf.float32)
    sigma_i = sigma_i**2
    sigma_i = tf.linalg.diag(sigma_i)
    mvn = tfd.MultivariateNormalTriL(loc=mu_i, scale_tril=tf.linalg.cholesky(sigma_i))
    # mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
    return mvn.prob([x_i])

def mdn_prediction(x_i, mu_i, sigma_i):
    x_i = tf.cast(x_i, tf.float32)
    x_i = tf.expand_dims(x_i, 1)
    mu_i = tf.expand_dims(mu_i, 0)
    sigma_i = sigma_i**2
    sigma_i = tf.linalg.diag(sigma_i)
    sigma_i = tf.expand_dims(sigma_i, 0)
    mvn = tfd.MultivariateNormalTriL(loc=mu_i, scale_tril=tf.linalg.cholesky(sigma_i))
    # mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
    return mvn.prob(x_i)

def gaussian_1d(x, mu, sigma):
    variance = sigma**2
    return 1 / (2.0 * np.pi * variance) * np.exp(-( (x-mu)**2 / (2.0 * variance)) )

def plot_gaussian(x, mu, sigma):
    y = gaussian_1d(x, mu, sigma)
    # y = np.random.normal(mu, sigma**2, len(x))
    plot_metric([x], [y], "Plot of Gaussian (mu={},sigma={})".format(mu, sigma), ["Gaussian dist."])

# def custom_training_and_predict_MDN():
#     optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     # tb_callback.set_model(nn_model)


#     loss = ml.MixtureDensityModelError(num_kernels=3)


#     @tf.function
#     def train_step(x, y):
#         ######### Your code starts here #########
#         # We want to perform a single training step (for one batch):
#         # 1. Make a forward pass through the model
#         # 2. Calculate the loss for the output of the forward pass
#         # 3. Based on the loss calculate the gradient for all weights
#         # 4. Run an optimization step on the weights.
#         # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
#         with tf.GradientTape() as tape:
#             # forward pass
#             y_est = nn_model(x, training=True) # use dropout
#             # compute the loss
#             current_loss = loss(y_est, y)
#         grads = tape.gradient(current_loss, nn_model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))
#         ########## Your code ends here ##########

#     train_loss(current_loss)

#     @tf.function
#     def train(train_data):
#         for x, y in train_data:
#             train_step(x, y)


#     train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

#     losses = []
#     for epoch in range(args.epochs):
#         # Reset the metrics at the start of the next epoch        
#         train_loss.reset_states()

#         train(train_data)

#         template = 'Epoch {}, Loss: {}'
#         print(template.format(epoch + 1, train_loss.result()))
#         losses.append(train_loss.result())

####################################################################################
####################################################################################
############################### Main  ##############################################
####################################################################################
    


def main():
    tf.compat.v1.enable_eager_execution()
    epochs = 1000
    # train_and_predict_basic_nn(epochs)
    # train_and_predict_forward_pass(epochs)
    train_and_predict_backward_pass(epochs)
    # train_and_predict_MDN(epochs)

main()