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

def train_and_predict_forward_pass(epochs):
    # dataset
    (x_train, y_train, x_test, y_test) = dg.generate_data_sinusoid()

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
    x = tf.linspace(0, 1, 100)
    y = model.predict(tf.transpose([x]))
    y1 = dg.sinusoid_random(x)
    plot_metric([x, x], [y, y1], "Plot of neural net approximating f(x)", ["nn(x)", "f(x)"])

####################################################################################
####################################################################################
############################### Main  ##############################################
####################################################################################
    
def main():
    epochs = 1000
    train_and_predict_forward_pass(epochs)



main()