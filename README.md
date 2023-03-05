# Exhibition of Mixed Density Networks

# 1. Note on Model Training

In short, a neural network (NN) can serve as function approximator. Here is a plot showing a basic NN approximating `sin(x)`:

![Approximating `sin(x)`](/images/01_2x64dense_softmax_10thou_ranged.png)

This model was trained on 100,000 samples of random, uniform input data (in the range `[-10,10]`). It contains two hidden layers of 64 units each, using a softmax activation function. The final output layer was linear. The loss function used was the Mean Squared Error. In summary:

| Hidden Layers | Activation | Samples     | Loss |
|--------------|--------------|-----------|------------|
| 2x64 | Softmax | 100,000      | MSE     |

Changing these hyperparameters can have a large impact on the ability of the NN to correctly approximate a function. There are some basic functions however which such a basic NN may fail to correctly approximate, despite experimentation with hyperparameters: where the outputs may be a distribution (i.e. a range of possible outputs).