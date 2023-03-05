#! env python
# Fundamental experiments with Mixed Density Networks

import tensorflow as tf
import models as ml
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

def train_basic_nn(epochs):
    (x_train, y_train, x_test, y_test) = generate_data_sinx()
    model = ml.BasicNN(x_train.shape[1], y_train.shape[1])
    loss = tf.keras.losses.MeanSquaredError()
    history = compile_and_train(model, loss, x_train, y_train, epochs)
    print("Evaluating model: " + model.name)
    return model.evaluate(x_test, y_test, verbose=2), model, history

def predict_basic_nn(model):
    x = tf.linspace(-5, 5, 100)
    y = model.predict(tf.transpose([x]))
    y1 = tf.math.sin(x)
    plot_metric([x, x], [y, y1], "Plot of neural net approximating sin (x)", ["nn(x)", "sin(x)"])


####################################################################################
####################################################################################
############################### Main  ##############################################
####################################################################################
    
def main():
    epochs = 5
    ((l, acc), model, history) = train_basic_nn(epochs)
    print("Achieved loss: {}, accuracy: {} ".format(l, acc))
    predict_basic_nn(model)


main()