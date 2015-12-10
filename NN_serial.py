
# This script does deep neural network inference without parallelization (numpy)
# for comparing speed and correctness
import numpy as np


# Generate nonlinear activation function
def ReLU(x):
    x[x < 0] = 0
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    out = np.zeros_like(x)
    for colm_i in range(x.shape[1]):        
        exp_elem = np.exp(x[:, colm_i])
        out[:, colm_i] = exp_elem / np.sum(exp_elem)
    return(out)

# Propagate inputs through network
def naive_dnn_serial(inputs, weights, n_layers, n_classes, n_neurons):
    layer_inputs = inputs
    for layer_i in range(n_layers - 1):
        # Apply sigmoid after each layer except for output layer
        # (maybe eventually use softmax)
        layer_inputs = weights[layer_i].dot(layer_inputs)

    return(layer_inputs)
