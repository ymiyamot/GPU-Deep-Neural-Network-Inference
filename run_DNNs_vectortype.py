from __future__ import division
import sys
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

import NN_serial


##### Function for default retrieval of gpu details #####
def setup_gpu():
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context([devices[2]])
    print 'This context is associated with ', len(context.devices), 'devices'
    
    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    return (queue, devices, context)

##### Function to generate input vectors to classify #####
def setup_inputs(input_sz, n_inputs, input_type, r_seed):
    np.random.seed(r_seed);
    if input_type == 'random':
        return(np.random.normal(size=(input_sz, n_inputs)).astype(np.float32))
    elif input_type == 'zeros':
        return(np.zeros(shape=(input_sz, n_inputs)).astype(np.float32))
    elif input_type == 'ones':
        return(np.ones(shape=(input_sz, n_inputs)).astype(np.float32))

##### Function to generate weights of neural network #####
def setup_weights(r_seed):
    np.random.seed(r_seed);
    weights = []
    weight_begin = []
    for layer_i in range(n_layers - 1):
        n_pre_layer = n_neurons[layer_i]
        n_post_layer = n_neurons[layer_i + 1]
        weights.append(np.random.normal(size=(n_post_layer, n_pre_layer))
                       .astype(np.float32))
        
        if layer_i == 0:
           weight_begin.append(np.int32(0))
        else:
           weight_begin.append(np.int32(weight_begin[-1]+(n_neurons[layer_i-1]*n_neurons[layer_i])))
        weights_1d = np.hstack([x.flatten() for x in weights])
    return (weights, weights_1d, weight_begin)

##### Function to print out diagnosis of DNN output (correctness) #####
def diagnose_performance(output_parallel, output_serial, print_extradeets_or_not):
    error_perc = 100 * np.abs(output_serial - output_parallel) / np.abs(output_serial)
    error = np.abs(output_serial - output_parallel)

    # Relative error
    print('Relative error')
    worst_ind = np.unravel_index(np.argmax(error_perc), output_parallel.shape)
    print('Max error of {}% at {}'.format(error_perc[worst_ind], worst_ind))
    print('Worst match: naive = {}, parallel = {}'
          .format(output_serial[worst_ind],
                  output_parallel[worst_ind]))
                  
    # Absolute error
    print('Absolute error')
    worst_ind = np.unravel_index(np.argmax(error), output_parallel.shape)
    print('Max error of {}% at {}'.format(error[worst_ind], worst_ind))
    print('Worst match: naive = {}, parallel = {}'
    .format(output_serial[worst_ind],
            output_parallel[worst_ind]))
            
    print('Outputs match? {}'.format(np.allclose(output_serial, output_parallel, rtol=0, atol=1e-4)))

    if print_extradeets_or_not:
        print('Inputs:')
        print(inputs)
        print('Weights:')
        print(weights)
        print('Outputs (dim: {})'.format(output_parallel.shape))
        print(output_parallel)
        print('Naive output (dim: {})'.format(output_serial.shape))
        print(output_serial)


##### Function to run our naive implementation of the DNN #####
def run_naive(inputs, weights_1d):
    # To run multiple inputs in the naive version, we will run one input vector at a time.
    mult_outputs = []
    layer_times = [] # Record how much time each layer takes
    for input_i in range(n_inputs):
        
        curr_input = inputs[:, input_i].astype(np.float32) # Take one input at a time
        
        # Transfer data to GPU format (4 is the number of bytes per float)
        gpu_neurons = cl.Buffer(context, cl.mem_flags.READ_WRITE, max(n_neurons) * 4)
        gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, (weights_1d.size) * 4)
        
        # Offload Kernel on GPU
        program = cl.Program(context, open('NN_naive.cl').read()).build(options='')
        
        # Send to the GPU, non-blocking
        cl.enqueue_copy(queue, gpu_neurons,  curr_input, is_blocking=False)
        cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)
        
        for layer_i in range(n_layers - 1):
            # For now, plan for each workgroup processing one row of weights
            local_size = (n_neurons[layer_i], 1)  # 64 pixels per work group
            global_size = tuple([n_neurons[layer_i], n_neurons[layer_i + 1]])
            
            # Local memory large enough to store one row of weights
            #gpu_local_memory = cl.LocalMemory(4 * (2 * n_neurons[layer_i]))
            gpu_summed_val = cl.LocalMemory(4 * n_neurons[layer_i])
            
            #print([n_neurons[layer_i], n_neurons[layer_i + 1], weight_begin[layer_i]])
            
            # Run Kernel on GPU
            event = program.NN_gpu_naive(queue, global_size, local_size,
                                         gpu_neurons,
                                         gpu_weights,
                                         gpu_summed_val,
                                         n_neurons[layer_i],
                                         weight_begin[layer_i])
            event.wait()
            seconds = (event.profile.end - event.profile.start) / 1e9
#            print("{} layer, {} seconds".format(layer_i, seconds))

            layer_times.append(seconds)

        # Post-processing
        out_neurons = np.zeros((max(n_neurons))).astype(np.float32)
        cl.enqueue_copy(queue, out_neurons, gpu_neurons, is_blocking=True)
        output_parallel = out_neurons[:n_classes]
        mult_outputs.append(output_parallel)
    return (np.vstack(mult_outputs).T, layer_times)

##### Function to run the blocked gpu implementation of the DNN #####
def run_blocked(inputs, weights_1d, n_inputs, n_neurons, local_sz, weight_begin):
    # Allocate GPU variables (4 is the number of bytes per float)
    gpu_inputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)
    gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, (weights_1d.size) * 4)
    gpu_outputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)

    # Offload Kernel on GPU
    program = cl.Program(context, open('NN_blocked.cl').read()).build(options='')
    
    # Send to the GPU, non-blocking (later, may need to load in chunks)
    cl.enqueue_copy(queue, gpu_inputs,  inputs, is_blocking=False)
    cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)
    
    # Run kernel
    layer_times = []
    for layer_i in range(n_layers - 1):
        # Set workgroup sizes and number of workers
        local_size = (local_sz, local_sz)  # 64 pixels per work group
        
        # Assume that this is a multiple of local_sz
        # !!!!!!!! GLOBAL SIZE SHOULD BE (X, Y), NOT (#ROWS, #COLUMNS) !!!!!!!!!
        global_size = (n_inputs, n_neurons[layer_i + 1]) # TODO: WHAT IS THE DOWNSIDE OF HAVING TOO MANY WORKERS HERE?
        assert global_size[0] % local_sz == 0 and global_size[1] % local_sz == 0
        
        # Allocate local memory
        gpu_local_inputs = cl.LocalMemory(4 * local_sz**2)
        gpu_local_weights = cl.LocalMemory(4 * local_sz**2)
        event = program.NN_gpu_blocked(queue, global_size, local_size,
                                       gpu_inputs,
                                       gpu_weights,
                                       gpu_outputs,
                                       gpu_local_inputs,
                                       gpu_local_weights,
                                       n_neurons[layer_i],
                                       n_inputs,
                                       n_neurons[layer_i + 1],
                                       weight_begin[layer_i])
        event.wait()
        seconds = (event.profile.end - event.profile.start) / 1e9
#        print("{} layer, {} seconds".format(layer_i, seconds))
        layer_times.append(seconds)
    
        gpu_inputs = gpu_outputs
    
    # Post-processing
    out_neurons = np.zeros((n_inputs * max(n_neurons))).astype(np.float32)
    cl.enqueue_copy(queue, out_neurons, gpu_outputs, is_blocking=True)
    output_parallel = out_neurons[:n_inputs * n_classes].reshape([n_classes, n_inputs])
    
    return(output_parallel, layer_times)

##### Function to run the vectorized + blocked gpu implementation of the DNN #####
def run_vectorized(inputs, weights_1d, n_inputs, n_neurons, local_sz, weight_begin):
    
    # Transfer inputs and weights to vector_type format
    # !!!!This needs to be able divide the input size!!!!
    vector_type_n = 4
    
    inputs_vec = np.zeros((inputs.shape[0], int(inputs.shape[1] / vector_type_n)),
                          dtype=cl_array.vec.float4)
        
    for input_y in range(inputs_vec.shape[0]):
        for input_x in range(inputs_vec.shape[1]):
            inputs_vec[input_y][input_x] = inputs[input_y,
                                                  range(input_x * vector_type_n,
                                                        (input_x + 1) * vector_type_n)]

    # Allocate GPU variables (4 is the number of bytes per float)
    gpu_inputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)
    gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, (weights_1d.size) * 4)
    gpu_outputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)
    
    # Offload Kernel on GPU
    program = cl.Program(context, open('NN_vectortype.cl').read()).build(options='')
    
    # Send to the GPU, non-blocking (later, may need to load in chunks)
    cl.enqueue_copy(queue, gpu_inputs,  inputs_vec, is_blocking=False)
    cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)
    
    # Run kernel
    layer_times = [] # Record how much time each layer takes
    for layer_i in range(n_layers - 1):
        # Set workgroup sizes and number of workers
        local_size = (int(local_sz / vector_type_n), local_sz)  # 64 pixels per work group
        
        # Assume that this is a multiple of local_sz
        
        # !!!!!!!! GLOBAL SIZE SHOULD BE (X, Y), NOT (#ROWS, #COLUMNS) !!!!!!!!!
        global_size = (int(n_inputs / vector_type_n),
                       n_neurons[layer_i + 1]) # TODO: WHAT IS THE DOWNSIDE OF HAVING TOO MANY WORKERS HERE?
        print('localsize = {}'.format(local_size))
        print('globalsize = {}'.format(global_size))
        assert global_size[0] % local_size[0] == 0 and global_size[1] % local_size[1] == 0

        # Allocate local memory
        gpu_local_inputs = cl.LocalMemory(4 * local_sz**2)
        gpu_local_weights = cl.LocalMemory(4 * local_sz**2)

        event = program.NN_gpu_vectortype(queue, global_size, local_size,
                                     gpu_inputs,
                                     gpu_weights,
                                     gpu_outputs,
                                     gpu_local_inputs,
                                     gpu_local_weights,
                                     n_neurons[layer_i],
                                     n_inputs,
                                     n_neurons[layer_i + 1],
                                     weight_begin[layer_i])
        event.wait()
        seconds = (event.profile.end - event.profile.start) / 1e9
#        print("{} layer, {} seconds".format(layer_i, seconds))
        layer_times.append(seconds)
        gpu_inputs = gpu_outputs

    # Post-processing
    out_neurons = np.zeros((n_inputs * max(n_neurons))).astype(np.float32)
    cl.enqueue_copy(queue, out_neurons, gpu_outputs, is_blocking=True)
    output_parallel = out_neurons[:n_inputs * n_classes].reshape([n_classes, n_inputs])
    
    return (output_parallel, layer_times)


##### Function to run the vectorized + blocked + forloop-rollout gpu implementation of the DNN #####
def run_vectorized_rollout(inputs, weights_1d, n_inputs, n_neurons, local_sz, weight_begin):
    
    # Transfer inputs and weights to vector_type format
    # !!!!This needs to be able divide the input size!!!!
    vector_type_n = 4
    
    # Convert the inputs into vectortype structure
    inputs_vec = np.zeros((inputs.shape[0], int(inputs.shape[1] / vector_type_n)),
                          dtype=cl_array.vec.float4)
    for input_y in range(inputs_vec.shape[0]):
        for input_x in range(inputs_vec.shape[1]):
            inputs_vec[input_y][input_x] = inputs[input_y,
                                                  range(input_x * vector_type_n,
                                                        (input_x + 1) * vector_type_n)]

    # Allocate GPU variables (4 is the number of bytes per float)
    gpu_inputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)
    gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, (weights_1d.size) * 4)
    gpu_outputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)
    
    # Offload Kernel on GPU
    program = cl.Program(context, open('NN_vectortype_forrollout.cl').read()).build(options='')
    
    # Send to the GPU, non-blocking (later, may need to load in chunks)
    cl.enqueue_copy(queue, gpu_inputs,  inputs_vec, is_blocking=False)
    cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)
    
    # Run kernel
    layer_times = [] # Record how much time each layer takes
    for layer_i in range(n_layers - 1):
        # Set workgroup sizes and number of workers
        local_size = (int(local_sz / vector_type_n), local_sz)  # 64 pixels per work group
        
        # Assume that this is a multiple of local_sz
        
        # !!!!!!!! GLOBAL SIZE SHOULD BE (X, Y), NOT (#ROWS, #COLUMNS) !!!!!!!!!
        global_size = (int(n_inputs / vector_type_n),
                       n_neurons[layer_i + 1]) # TODO: WHAT IS THE DOWNSIDE OF HAVING TOO MANY WORKERS HERE?
        print('localsize = {}'.format(local_size))
        print('globalsize = {}'.format(global_size))
        assert global_size[0] % local_size[0] == 0 and global_size[1] % local_size[1] == 0

        # Allocate local memory
        gpu_local_inputs = cl.LocalMemory(4 * local_sz**2)
        gpu_local_weights = cl.LocalMemory(4 * local_sz**2)

        event = program.NN_gpu_vectortype_forrollout(queue, global_size, local_size,
                                        gpu_inputs,
                                        gpu_weights,
                                        gpu_outputs,
                                        gpu_local_inputs,
                                        gpu_local_weights,
                                        n_neurons[layer_i],
                                        n_inputs,
                                        n_neurons[layer_i + 1],
                                        weight_begin[layer_i])
        event.wait()
        seconds = (event.profile.end - event.profile.start) / 1e9
#        print("{} layer, {} seconds".format(layer_i, seconds))
        layer_times.append(seconds)
        gpu_inputs = gpu_outputs

    # Post-processing
    out_neurons = np.zeros((n_inputs * max(n_neurons))).astype(np.float32)
    cl.enqueue_copy(queue, out_neurons, gpu_outputs, is_blocking=True)
    output_parallel = out_neurons[:n_inputs * n_classes].reshape([n_classes, n_inputs])

    return (output_parallel, layer_times)



if __name__ == '__main__':
    [queue, devices, context] = setup_gpu()

#    r_seed = np.random.random_integers(1, 120)
    r_seed = 54
    print('Seed is: {}'.format(r_seed))
    np.random.seed(r_seed);

    ########## Initialization ##########
    # Set up neural network parameters
    # Decide the parameters of the structure of the neural network

    # Large inputs for testing
#    n_layers = np.int32(4) # Including input and output layer
#    n_inputs = np.int32(2**8)
#    input_sz = np.int32(2**10) # Basic MNIST data input size
#    n_classes = np.int32(2**5) # Size of output layer
#    layer_sz = np.int32(2**10)
#    local_sz = 2**5

# Simple inputs for debugging
#    n_layers = np.int32(2) # Including input and output layer
#    n_inputs = np.int32(2**2)
#    input_sz = np.int32(2**3) # Basic MNIST data input size
#    n_classes = np.int32(2**3) # Size of output layer
#    layer_sz = np.int32(2**3)
#    local_sz = 2**2

    n_layers = np.int32(10) # Including input and output layer
    n_inputs = np.int32(2**2)
    input_sz = np.int32(2**3) # Basic MNIST data input size
    n_classes = np.int32(2**3) # Size of output layer
    layer_sz = np.int32(2**3)
    local_sz = 2**2
    
    print('\n\n**************** DNN architecture: ****************')
    print('# of layers: {} (including input and output layer)'.format(n_layers))
    print('# of inputs: {}'.format(n_inputs))
    print('Output classes: {}'.format(n_classes))
    print('Input vector size: {}'.format(input_sz))
    print('Layer sizes: {}'.format(layer_sz))
    print('Local size:{}'.format(local_sz))
    print('*******************************************************\n\n')
    
    

#    n_layers = np.int32(2) # Including input and output layer
#    n_inputs = np.int32(2**5)
#    input_sz = np.int32(2**8) # Basic MNIST data input size
#    n_classes = np.int32(2**6) # Size of output layer
#    layer_sz = np.int32(2**6)
#    local_sz = 2**5

    n_neurons = [input_sz] + [layer_sz] * (n_layers - 2) + [n_classes]


    # Generate weights
    # weights: list where each element corresponds to a matrix of weights for one layer
    # weights_1d : All vectorized weights
    # weight_begin : Start weight locations for each layer computation
    [weights, weights_1d, weight_begin] = setup_weights(r_seed)

    # Generate inputs
    inputs = setup_inputs(input_sz, n_inputs, 'random', r_seed) # random inputs

    ########## Serial implementation of DNN ##########
    output_serial = NN_serial.naive_dnn_serial(inputs,
                                               weights,
                                               n_layers,
                                               n_classes,
                                               n_neurons)
#    print("Serial outputs (run on cpu) : \n{}".format(output_serial))
    ####################################

    ########## Naive implementation of DNN ##########
    print('\n\n================== Executing Naive implementation ==================')
    [output_naive, times_naive] = run_naive(inputs, weights_1d)
#    diagnose_performance(output_naive, output_serial, False)
    print('Total time: {}'.format(sum(times_naive)))
    
    ########## Vectorized + Blocked implementation of DNN ##########
    print('\n\n================== Executing Blocked implementation ==================')
    [output_blocked, times_blocked] = run_blocked(inputs,
                                                  weights_1d,
                                                  n_inputs,
                                                  n_neurons,
                                                  local_sz,
                                                  weight_begin)
    diagnose_performance(output_blocked, output_naive, True)
    print('Total time: {}'.format(sum(times_blocked)))
    print('Improvement over naive: {}x'.format(sum(times_naive) / sum(times_blocked)))
    print(output_serial)

#    ########## Vectorized + Blocked implementation of DNN ##########
#    print('\n\n================== Executing Vectorized implementation ==================')
#    [output_vectorized, times_vectorized] = run_vectorized(inputs,
#                                                           weights_1d,
#                                                           n_inputs,
#                                                           n_neurons,
#                                                           local_sz,
#                                                           weight_begin)
#    diagnose_performance(output_vectorized, output_naive, False)
#    print('Total time: {}'.format(sum(times_vectorized)))
#    print('Improvement over naive: {}x'.format(sum(times_naive) / sum(times_vectorized)))
#    
#    ########## Vectorized + Blocked + loop-rollout implementation of DNN ##########
#    print('\n\n================== Executing Rollout implementation ==================')
#    [output_rollout, times_rollout] = run_vectorized_rollout(inputs,
#                                                             weights_1d,
#                                                             n_inputs,
#                                                             n_neurons,
#                                                             local_sz,
#                                                             weight_begin)
#    diagnose_performance(output_rollout, output_naive, False)
#    print('Total time: {}'.format(sum(times_rollout)))
#    print('Improvement over naive: {}x'.format(sum(times_naive) / sum(times_rollout)))
#
#


