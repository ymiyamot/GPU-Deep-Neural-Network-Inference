
# coding: utf-8

# In[1]:

# This script does deep neural network inference without parallelization
# for comparing speed and correctness
import numpy as np


# In[8]:

# Decide the parameters of the structure of the neural network
n_layers = 4 # Including input and output layer
n_classes = 2 # Size of output layer
n_neurons = [2**8, 2**8, 2**8, n_classes]


# In[27]:

# Generate weights
weights = []
for layer_i in range(n_layers - 1):
    n_pre_layer = n_neurons[layer_i]
    n_post_layer = n_neurons[layer_i + 1]
    weights.append(np.random.normal(size=(n_post_layer, n_pre_layer)))


# In[81]:

# Generate inputs
n_inputs = 3
inputs = np.random.normal(size=(n_neurons[0], n_inputs)) # random inputs
# inputs = np.zeros(shape=(n_neurons[0], n_inputs)) # zero inputs 


# In[82]:

# Generate nonlinear activation function
def nonlin_activation(x):
    return 2 * np.exp(x) / (np.exp(x) + 1) - 1

def softmax(x):
    out = np.zeros_like(x)
    for colm_i in range(x.shape[1]):        
        exp_elem = np.exp(x[:, colm_i])
        out[:, colm_i] = exp_elem / np.sum(exp_elem)
    return(out)


# In[83]:

# Propagate inputs through network
def infer_np_serial(inputs):
    layer_inputs = inputs
    for layer_i in range(n_layers - 1):
        if layer_i != n_layers - 2:
            layer_inputs = nonlin_activation(weights[layer_i].dot(layer_inputs))
        else:
            layer_inputs = softmax(weights[layer_i].dot(layer_inputs))
            output = layer_inputs
    return(output)

