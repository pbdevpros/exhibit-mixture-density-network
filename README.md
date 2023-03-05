# Exhibition of Mixed Density Networks

# 1. Note on Model Training

In short, neural networks (NN) serve as function approximators. Here is a plot showing a basic NN approximating `sin(x)`:

![Approximating `sin(x)`](/images/01_2x64dense_softmax_10thou_ranged.png)

This model was trained on 100,000 samples of random, uniform input data (in the range `[-10,10]`). It contains two hidden layers of 64 units each, using a softmax activation function. The final output layer was linear. The loss function used was the Mean Squared Error.

| Hidden Layers | Samples     | Loss |
|--------------|-----------|------------|
| 2x64 | 100,000      | **MSE**      |