from __future__ import division
import sys
import pyopencl as cl
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
    n_layers = np.int32(5) # Including input and output layer
    n_inputs = np.int32(1)
    input_sz = np.int32(8)
    n_classes = np.int32(2) # Size of output layer
    layer_sz = np.int32(2**4)
    n_neurons = [input_sz, layer_sz, layer_sz, layer_sz, n_classes]

    ### Initlization ###
    # Generate weights
    # weights_1d : All vectorized weights
    # weight_start : Start weight locations for each layer computation
    weights = []
    weight_start = []
    for layer_i in range(n_layers - 1):
        n_pre_layer = n_neurons[layer_i]
        n_post_layer = n_neurons[layer_i + 1]
#        weights.append(np.random.normal(size=(n_post_layer, n_pre_layer))
#                       .astype(np.float32))
        weights.append(np.ones(shape=(n_post_layer, n_pre_layer))
                       .astype(np.float32))
        if layer_i == 0:
            weight_start.append(np.int32(0))
        else:
            weight_start.append(np.int32(weight_start[-1]+n_neurons[layer_i-1]*n_neurons[layer_i]))
    weights_1d = np.hstack([x.flatten() for x in weights])

    # Generate inputs
    # random inputs
#    inputs = np.random.normal(size=(n_neurons[0], n_inputs)).astype(np.float32)
#     inputs = np.zeros(shape=(n_neurons[0], n_inputs)).astype(np.float32) # zero inputs
    inputs = 2 * np.ones(shape=(input_sz, n_inputs)).astype(np.float32) # one inputs

    ### Serial implementation of DNN ### 
    output_serial = NN_serial.naive_dnn_serial(inputs,
                                               weights,
                                               n_layers,
                                               n_classes,
                                               n_neurons)
    #################################### 


    ### Parallel implementation of DNN ### 
    # Transfer data to GPU format (4 is the number of bytes per float)
    gpu_neurons = cl.Buffer(context, cl.mem_flags.READ_WRITE, max(n_neurons) * 4)
    gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, (weights_1d.size) * 4)
    
    # Offload Kernel on GPU
    program = cl.Program(context, open('NN_naive.cl').read()).build(options='')
    
    # Send to the GPU, non-blocking (WHAT IS BLOCKING?)
    cl.enqueue_copy(queue, gpu_neurons,  inputs, is_blocking=False)
    cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)

    for layer_i in range(n_layers - 1):
        # For now, plan for each workgroup processing one row of weights
        local_size = (n_neurons[layer_i], 1)  # 64 pixels per work group
        global_size = tuple([n_neurons[layer_i], n_neurons[layer_i + 1]])

        # Local memory large enough to store one row of weights
        #gpu_local_memory = cl.LocalMemory(4 * (2 * n_neurons[layer_i]))
        gpu_summed_val = cl.LocalMemory(4 * n_neurons[layer_i])

        #print([n_neurons[layer_i], n_neurons[layer_i + 1], weight_start[layer_i]])
        
        # Run Kernel on GPU
        event = program.NN_gpu_naive(queue, global_size, local_size,
                             gpu_neurons,
                             gpu_weights,
                             gpu_summed_val,
                             n_neurons[layer_i],
                             weight_start[layer_i])
        event.wait()
        seconds = (event.profile.end - event.profile.start) / 1e9
        print("{} layer, {} seconds".format(layer_i, seconds))
    
    # Post-processing
    out_neurons = np.zeros((max(n_neurons))).astype(np.float32)
    cl.enqueue_copy(queue, out_neurons, gpu_neurons, is_blocking=True)
    output_parallel = out_neurons[:n_classes]
    print("Serial outputs (run on cpu) : {}".format(output_serial))
    print("Parallel outputs (run on gpu) : {}".format(output_parallel))
