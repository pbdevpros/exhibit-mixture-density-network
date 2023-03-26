# Exhibition of Mixture Density Networks

## Overview 
The below is an explanation and discussion of the paper on [Mixed Density Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf) from C. M. Bishop. The results of the paper have been reproduced in the associated source code. 

## Modeling Outputs as Distributions
### Introduction

As is often mentioned, a neural network (NN) can serve as function approximator. There are some basic functions however where a NN may fail to correctly approximate (even despite experimentation with hyperparameters): where the outputs may be a distribution (i.e. a range of possible outputs).

### Classic Neural Networks

To clarify the above statement, take the example of a function which includes a random variable:

```python
f(x) = x + 0.3 * sin(2*pi*x) + E
```

where E is a random variable with distribution betwee (-0.1, 0.1). Calculating `f(x)` here is an example of a "forward-problem", where we are predicting the output given an input (`x`). The NN can provide an excellent represtation of the underlying generator function (`f(x)`). Following the paper, a network of 1 hidden layer with 4 units with `tanh` activation functions, trained on 1,000 samples (evenly spaced between 0 and 1) for 1,000 epochs can be used to represent the function. (Note: the Adam optimizer was used here, whereas in the paper the BFGS optimization algorithm is used).

![Approximating `f(x)` with Bishop NN](/images/04_1x4tanh_fx_1000samples_1000epochs.png)

<div align="center">  

| Hidden Layers | Hidden Layers Depth | Activation | Samples     | Loss | Epochs |
|:--------------:|:--------------:|:-----------:|:------------:|:------------:|:------------:|
|1|[ 4 ]|tanh| 1,000      | MSE     | 1,000|  

</div>

### Mixture Density Networks

We can now consider the "inverse" problem, of predicting the input `x` based on the output `f(x)`. This is essentially flipping the inputs/outputs to the NN and observing if the model can approximate this inverse function. The only required change to the network is the loss function - where a mixture of Gaussians are used. Implementing the network as in the paper (where 3 Gaussian kernels are used for the network), we can see the network can predict the results of the "inverse" problem quite well:

![Approximating `x` - MDN](/images/08_MDN_prediction.png)