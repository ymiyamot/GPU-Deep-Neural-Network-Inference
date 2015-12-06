from __future__ import division
import sys
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

import NN_serial

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

if __name__ == '__main__':
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
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    ### Set up neural network parameters ###
    # Decide the parameters of the structure of the neural network
#    n_layers = np.int32(3) # Including input and output layer
#    n_inputs = np.int32(256)
#    input_sz = np.int32(128) # Basic MNIST data input size
#    n_classes = np.int32(32) # Size of output layer
#    layer_sz = np.int32(2**10)

# Large inputs
#    n_layers = np.int32(2) # Including input and output layer
#    n_inputs = np.int32(2**5)
#    input_sz = np.int32(2**6) # Basic MNIST data input size
#    n_classes = np.int32(2**6) # Size of output layer
#    layer_sz = np.int32(2**6)
#    local_sz = 2**5

# Simple inputs for debugging
    n_layers = np.int32(2) # Including input and output layer
    n_inputs = np.int32(2**2)
    input_sz = np.int32(2**3) # Basic MNIST data input size
    n_classes = np.int32(2**3) # Size of output layer
    layer_sz = np.int32(2**3)
    local_sz = 2**2

#    n_layers = np.int32(3) # Including input and output layer
#    n_inputs = np.int32(2)
#    input_sz = np.int32(4) # Basic MNIST data input size
#    n_classes = np.int32(2) # Size of output layer
#    layer_sz = np.int32(2**2)
    n_neurons = [input_sz] + [layer_sz] * (n_layers - 2) + [n_classes]

    ### Initialization ###
    # Generate weights
    # weights_1d : All vectorized weights
    # weight_begin : Start weight locations for each layer computation
    weights = []
    weight_begin = []
    for layer_i in range(n_layers - 1):
        n_pre_layer = n_neurons[layer_i]
        n_post_layer = n_neurons[layer_i + 1]
        weights.append(np.random.normal(size=(n_post_layer, n_pre_layer))
                       .astype(np.float32))
#        weights.append(np.ones(shape=(n_post_layer, n_pre_layer))
#                       .astype(np.float32))
        if layer_i == 0:
            weight_begin.append(np.int32(0))
        else:
            weight_begin.append(np.int32(weight_begin[-1]+(n_neurons[layer_i-1]*n_neurons[layer_i])))
    weights_1d = np.hstack([x.flatten() for x in weights])

    # Generate inputs
    # random inputs
    inputs = np.random.normal(size=(input_sz, n_inputs)).astype(np.float32)
#    inputs = np.round(100 * np.random.normal(size=(input_sz, n_inputs)).astype(np.float32))
#    inputs = np.array([[ 0.13862385, -1.04301488],
#                       [ 0.6325286 ,  0.64799154],
#                       [-0.61446768,  0.59548354],
#                       [ 0.30874914,  0.42244878]]).astype(np.float32)

#    inputs = np.zeros(shape=(input_sz, n_inputs)).astype(np.float32) # zero inputs
#    inputs = 3 * np.ones(shape=(input_sz, n_inputs)).astype(np.float32) # one inputs

    ### Serial implementation of DNN ###
#    output_serial= []
#    for input_i in range(n_inputs):
#        output_serial.append(NN_serial.naive_dnn_serial(inputs[:, input_i],
#                                                        weights,
#                                                        n_layers,
#                                                        n_classes,
#                                                        n_neurons))
#    print("Serial outputs (run on cpu) : \n{}".format(np.vstack(output_serial).T))

    output_serial = NN_serial.naive_dnn_serial(inputs,
                                               weights,
                                               n_layers,
                                               n_classes,
                                               n_neurons)
    print("Serial outputs (run on cpu) : \n{}".format(output_serial))
    ####################################

    weight_begin_vectortype = [int(x / 4) for x in weight_begin]
    #print(weight_begin)
    #print(weight_begin_vectortype)
    
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

    weights_1d_vec = np.zeros(int(weights_1d.size / vector_type_n),
                          dtype=cl_array.vec.float4)

    weights_vectype = []
    for layer_i in range(n_layers - 1):
        cur_weights = np.zeros((n_neurons[layer_i + 1] / vector_type_n,
                               n_neurons[layer_i])).astype(cl_array.vec.float4)
        for colm_i in range(n_neurons[layer_i]):
            for row_i in range(int(n_neurons[layer_i + 1] / vector_type_n)):
                cur_weights[row_i, colm_i] = weights[layer_i][np.arange(vector_type_n)
                                                               + vector_type_n * row_i, colm_i]
        weights_vectype.append(cur_weights)
    weights_vectype_1d = np.hstack([x.flatten() for x in weights_vectype])


    # Allocate GPU variables (4 is the number of bytes per float)
    gpu_inputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)
    gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, (weights_1d.size) * 4)
    gpu_outputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, n_inputs * max(n_neurons) * 4)

    # Offload Kernel on GPU
    program = cl.Program(context, open('NN_vectortype.cl').read()).build(options='')

    # Send to the GPU, non-blocking (later, may need to load in chunks)
    cl.enqueue_copy(queue, gpu_inputs,  inputs_vec, is_blocking=False)
    cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)
    #cl.enqueue_copy(queue, gpu_weights, weights_vectype_1d, is_blocking=False)

    # Run kernel
    for layer_i in range(n_layers - 1):
        # Set workgroup sizes and number of workers
        local_size = (local_sz, local_sz)
        
        # Assume that this is a multiple of local_sz

        # !!!!!!!! GLOBAL SIZE SHOULD BE (X, Y), NOT (#ROWS, #COLUMNS) !!!!!!!!!
        global_size = (n_inputs, n_neurons[layer_i + 1]) # TODO: WHAT IS THE DOWNSIDE OF HAVING TOO MANY WORKERS HERE?
        assert global_size[0] % local_sz == 0 and global_size[1] % local_sz == 0
        
        print('localsize = {}'.format(local_sz))
        print('globalsize = {}'.format(global_size))
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
        print("{} layer, {} seconds".format(layer_i, seconds))

        gpu_inputs = gpu_outputs

    # Post-processing
    out_neurons = np.zeros((n_inputs * max(n_neurons))).astype(np.float32)
    cl.enqueue_copy(queue, out_neurons, gpu_outputs, is_blocking=True)
    output_parallel = out_neurons[:n_inputs * n_classes].reshape([n_classes, n_inputs])
    print(output_serial)
    print(out_neurons)
    print(inputs)
    print(weights)
    #test_input = inputs[0:4,0:4]
    #test_weights = weights[0][0:4,0:4]
    #test_input = inputs[4:8,0:4]
    #test_weights = weights[0][0:4,4:8]
    #test_input = inputs
    #test_weights = weights[0][0:4,0:8]
    test_input = inputs
    test_weights = weights[0][4:8,0:8]
    acc = np.zeros(4)
    for i in range(8):
        acc += np.multiply(test_input[i,:],test_weights[:,i]) 
    print acc
#    print('Outputs match? {}'.format(np.allclose(output_serial.flatten(), out_neurons[:n_inputs * n_classes])))

#    print(output_serial.shape)
#    print(output_parallel)
#    print(output_parallel.shape)
#    print("Parallel outputs (run on gpu) : \n{}".format(np.vstack(mult_outputs).T))
#    print(100 * np.abs(output_serial - output_parallel) / np.abs(output_serial))
    print('Outputs match? {}'.format(np.allclose(output_serial, output_parallel, rtol=1e-03)))

#    ### Parallel implementation of DNN ###
#
#    # To run multiple inputs in the naive version, we will run one input vector at a time.
#    mult_outputs = []
#    for input_i in range(n_inputs):
#
#        curr_input = inputs[:, input_i].astype(np.float32) # Take one input at a time
#        
#        # Transfer data to GPU format (4 is the number of bytes per float)
#        gpu_neurons = cl.Buffer(context, cl.mem_flags.READ_WRITE, max(n_neurons) * 4)
#        gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, (weights_1d.size) * 4)
#        
#        # Offload Kernel on GPU
#        program = cl.Program(context, open('NN_naive.cl').read()).build(options='')
#        
#        # Send to the GPU, non-blocking
#        cl.enqueue_copy(queue, gpu_neurons,  curr_input, is_blocking=False)
#        cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)
#
#        for layer_i in range(n_layers - 1):
#            # For now, plan for each workgroup processing one row of weights
#            local_size = (n_neurons[layer_i], 1)  # 64 pixels per work group
#            global_size = tuple([n_neurons[layer_i], n_neurons[layer_i + 1]])
#
#            # Local memory large enough to store one row of weights
#            #gpu_local_memory = cl.LocalMemory(4 * (2 * n_neurons[layer_i]))
#            gpu_summed_val = cl.LocalMemory(4 * n_neurons[layer_i])
#
#            #print([n_neurons[layer_i], n_neurons[layer_i + 1], weight_begin[layer_i]])
#            
#            # Run Kernel on GPU
#            event = program.NN_gpu_naive(queue, global_size, local_size,
#                                 gpu_neurons,
#                                 gpu_weights,
#                                 gpu_summed_val,
#                                 n_neurons[layer_i],
#                                 weight_begin[layer_i])
#            event.wait()
#            seconds = (event.profile.end - event.profile.start) / 1e9
#            print("{} layer, {} seconds".format(layer_i, seconds))
#
##        out_neurons = np.zeros((max(n_neurons))).astype(np.float32)
##        cl.enqueue_copy(queue, out_neurons, gpu_neurons, is_blocking=True)
##        print("Intermediate outputs (run on gpu) : {}".format(out_neurons))
#
#        # Post-processing
#        out_neurons = np.zeros((max(n_neurons))).astype(np.float32)
#        cl.enqueue_copy(queue, out_neurons, gpu_neurons, is_blocking=True)
#        output_parallel = out_neurons[:n_classes]
#        mult_outputs.append(output_parallel)
#
##    print("Serial outputs (run on cpu) : \n{}".format(output_serial))
##    print("Parallel outputs (run on gpu) : \n{}".format(np.vstack(mult_outputs).T))
#    print('Outputs match? {}'.format(np.allclose(output_serial, np.vstack(mult_outputs).T)))

