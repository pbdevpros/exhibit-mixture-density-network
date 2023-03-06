# Exhibition of Mixed Density Networks

# 1. Note on Model Training

In short, a neural network (NN) can serve as function approximator. Here is a plot showing a basic NN approximating `sin(x)`:

![Approximating `sin(x)`](/images/01_2x64dense_softmax_100thou_ranged.png)

This model was trained on 100,000 samples of random, uniform input data (in the range `[-10,10]`). It contains two hidden layers of 64 units each, using a softmax activation function. The final output layer was linear. The loss function used was the Mean Squared Error. In summary:

| Hidden Layers | Hidden Layers Depth | Activation | Samples     | Loss | Epochs |
|--------------|--------------|-----------|------------|------------|------------|
| 2 | [ 64, 64] | Softmax | 100,000      | MSE     | 5 |

Changing these hyperparameters can have a large impact on the ability of the NN to correctly approximate a function. There are some basic functions however which such a basic NN may fail to correctly approximate, despite experimentation with hyperparameters: where the outputs may be a distribution (i.e. a range of possible outputs).

# 2. Modeling Outputs as Distributions

To clarify the above statement, take the example of a function which includes a random variable:

```python
f(x) = x + 0.3 * sin(2*pi*x) + E
```

where E is a random variable with distribution betwee (-0.1, 0.1). Calculating `f(x)` here is an example of a "forward-problem", where we are predicting the output given an input (`x`). The NN can provide an excellent represtation of the underlying generator function (`f(x)`). Using the same network and hyperparameters, we can see a clear mapping of the function (averaging the noise):

![Approximating `f(x)`](/images/05_2x64softmax_fx_1000samples_1000epochs.png)

In fact, a much smaller network can achieve the same output. Following the paper, a network of 1 hidden layer with 4 units with `tanh` activation functions, trained on 1,000 samples (evenly spaced between 0 and 1) for 1,000 epochs. (Note: the Adam optimizer is used here, whereas in the paper the BFGS optimization algorithm is used).

![Approximating `f(x)` with Bishop NN](/images/04_1x4tanh_fx_1000samples_1000epochs.png)

| Hidden Layers | Hidden Layers Depth | Activation | Samples     | Loss | Epochs |
|--------------|--------------|-----------|------------|------------|------------|
| 1 | [ 4 ] | tanh | 1,000      | MSE     | 1,000|

We can now consider the "inverse" problem, of predicting the input `x` based on the output `f(x)`. This is essentially flipping the inputs/outputs to the NN and observing if the model can approximated this inverse function. Increasing the depth of the model to 20 units in the hidden layer (the optimally tuned recommendation by Bishop, et. al), it is still not possible (note these results do not replicate Bishop, however it is clear from the paper it is not possible): 


![Approximating `x` - backward pass](/images/06_1x20tanh_1000samples_1000epochs.png)