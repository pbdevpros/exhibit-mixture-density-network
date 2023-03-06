#! env python
# Fundamental experiments with Mixed Density Networks

import tensorflow as tf
import models as ml
import data_generator as dg
from data_generator import generate_data_sinx
from utilities import save_history, load_history, plot_histories, plot_xy, plot_metric

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
    x = tf.linspace(0, 1, 100)
    y1 = dg.sinusoid_random(x)
    y = model.predict(tf.transpose([x]))
    plot_metric([y, y1], [x, x], "Plot of neural net approximating f(x)", ["nn(x)", "f(x)"])

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
    x = tf.linspace(0, 1, 100)
    y1 = dg.sinusoid_random(x)
    y = model.predict(tf.transpose([x]))
    plot_metric([y, y1], [x, x], "Plot of neural net approximating f(x)", ["nn(x)", "f(x)"])

####################################################################################
####################################################################################
############################### Main  ##############################################
####################################################################################
    


def main():
    tf.compat.v1.enable_eager_execution()
    epochs = 3
    # train_and_predict_basic_nn(epochs)
    # train_and_predict_forward_pass(epochs)
    # train_and_predict_backward_pass(epochs)
    train_and_predict_MDN(epochs)

main()