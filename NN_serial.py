
# This script does deep neural network inference without parallelization (numpy)
# for comparing speed and correctness
import numpy as np


# Generate nonlinear activation function
def nonlin_activation(x):
    return 2 * np.exp(x) / (np.exp(x) + 1) - 1

def softmax(x):
    out = np.zeros_like(x)
    for colm_i in range(x.shape[1]):        
        exp_elem = np.exp(x[:, colm_i])
        out[:, colm_i] = exp_elem / np.sum(exp_elem)
    return(out)

# Propagate inputs through network
def infer_np_serial(inputs, weights, n_layers, n_classes, n_neurons):
    layer_inputs = inputs
    for layer_i in range(n_layers - 1):
        if layer_i != n_layers - 2:
            layer_inputs = nonlin_activation(weights[layer_i].dot(layer_inputs))
        else:
            layer_inputs = softmax(weights[layer_i].dot(layer_inputs))
            output = layer_inputs
    return(output)
