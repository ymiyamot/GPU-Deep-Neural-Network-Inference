
# This script does deep neural network inference without parallelization (numpy)
# for comparing speed and correctness
import numpy as np


# Generate nonlinear activation function
def ReLU(x):
    x[np.where( x < 0.0 )] = 0
    return x

def sigmoid(x):
    return 2 * np.exp(x) / (np.exp(x) + 1) - 1

def softmax(x):
    out = np.zeros_like(x)
    for colm_i in range(x.shape[1]):        
        exp_elem = np.exp(x[:, colm_i])
        out[:, colm_i] = exp_elem / np.sum(exp_elem)
    return(out)

# Propagate inputs through network
def naive_dnn_serial(inputs, weights, n_layers, n_classes, n_neurons):
    layer_inputs = inputs
    print(type(inputs[0]))
    for layer_i in range(n_layers - 1):
#        layer_inputs = ReLU(weights[layer_i].dot(layer_inputs))
        layer_inputs = weights[layer_i].dot(layer_inputs)
#        print(layer_inputs.shape)
        print(type(layer_inputs[0]))
    output = layer_inputs
#    return(output.flatten())

    print(type(output[0]))
    return(output)
# Propagate inputs through network
def infer_np_serial(inputs, weights, n_layers, n_classes, n_neurons):
    layer_inputs = inputs
    for layer_i in range(n_layers - 1):
        if layer_i != n_layers - 2:
            layer_inputs = sigmoid(weights[layer_i].dot(layer_inputs))
        else:
            layer_inputs = softmax(weights[layer_i].dot(layer_inputs))
            output = layer_inputs
    return(output)
