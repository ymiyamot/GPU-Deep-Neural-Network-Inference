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
    context = cl.Context([devices[2]])
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name



    ### Set up neural network parameters ###
    # Decide the parameters of the structure of the neural network
    n_layers = np.int32(4) # Including input and output layer
    n_classes = np.int32(2) # Size of output layer
#    layer_sz = np.int32(2**8)
    layer_sz = np.int32(2**2)
    n_neurons = [layer_sz, layer_sz, layer_sz, n_classes]

    # Generate weights
    weights = []
    for layer_i in range(n_layers - 1):
        n_pre_layer = n_neurons[layer_i]
        n_post_layer = n_neurons[layer_i + 1]
#        weights.append(np.random.normal(size=(n_post_layer, n_pre_layer))
#                       .astype(np.float32))
        weights.append(np.ones(shape=(n_post_layer, n_pre_layer))
                       .astype(np.float32))

    # Generate inputs
    n_inputs = np.int32(3)
    # random inputs
#    inputs = np.random.normal(size=(n_neurons[0], n_inputs)).astype(np.float32)
    # inputs = np.zeros(shape=(n_neurons[0], n_inputs)).astype(np.float32) # zero inputs
    inputs = np.ones(shape=(n_neurons[0], n_inputs)).astype(np.float32) # one inputs

    output_serial = NN_serial.infer_np_serial(inputs,
                                    weights,
                                    n_layers,
                                    n_classes,
                                    n_neurons)
    print(inputs)
    print(weights)
    print(output_serial)

    output_parallel = np.zeros_like(output_serial)


    weights_1d = np.hstack([x.flatten() for x in weights])

    # Pointer to counter that keeps track of how many computations were performed
    computation_count = np.zeros(1).astype(np.int32)

    # Transfer data to GPU format (4 is the number of bytes per float)
    gpu_inputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, inputs.size * 4)
    gpu_weights = cl.Buffer(context, cl.mem_flags.READ_ONLY, weights_1d.size * 4)
    gpu_output = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, n_inputs * n_classes * 4)
    gpu_computation_count = cl.Buffer(context, cl.mem_flags.READ_WRITE, 4)

    # Probably get rid of this later
    gpu_interm_inputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, inputs.size * 4)
    gpu_tmp_inputs = cl.Buffer(context, cl.mem_flags.READ_WRITE, inputs.size * 4)

    program = cl.Program(context, open('NN_parallel.cl').read()).build(options='')

    # Send to the device, non-blocking (WHAT IS BLOCKING?)
    cl.enqueue_copy(queue, gpu_inputs, inputs, is_blocking=False)
    cl.enqueue_copy(queue, gpu_weights, weights_1d, is_blocking=False)
    computation_count[0] = 0
    cl.enqueue_copy(queue, gpu_computation_count, computation_count, is_blocking=False)

    # For now, plan for each workgroup processing one row of weights
    local_size = (64, 1)  # 64 pixels per work group
    global_size = tuple([64, layer_sz])
    print global_size
    
    # Local memory large enough to store one row of weights
    gpu_local_memory = cl.LocalMemory(4 * layer_sz)

    program.NN_gpu_naive(queue, global_size, local_size, gpu_inputs,
                         gpu_weights, gpu_local_memory, gpu_output,
                         gpu_interm_inputs, gpu_tmp_inputs,
                         gpu_computation_count,
                         n_inputs, n_layers, n_classes, layer_sz)

    # What does blocking mean???
    cl.enqueue_copy(queue, output_parallel, gpu_output, is_blocking=True)
    print(output_parallel)

    interm_inputs = np.zeros_like(inputs)
    cl.enqueue_copy(queue, interm_inputs, gpu_interm_inputs, is_blocking=True)
    print(interm_inputs)

    cl.enqueue_copy(queue, computation_count,
                    gpu_computation_count, is_blocking=True)
    print(computation_count)


#    cl.enqueue_copy(queue, gpu_image_a, host_image, is_blocking=False)
#    width = np.int32(host_image.shape[1])
#    height = np.int32(host_image.shape[0])
#    halo = np.int32(1)
#
#    # Create a local memory per working group that is
#    # the size of an int (4 bytes) * (N+2) * (N+2), where N is the local_size
#    buf_size = (np.int32(local_size[0] + 2 * halo), np.int32(local_size[1] + 2 * halo))
#    gpu_local_memory = cl.LocalMemory(4 * buf_size[0] * buf_size[1])
#
#    # initialize labels
#    program.initialize_labels(queue, global_size, local_size,
#                              gpu_image, gpu_labels,
#                              width, height)
#
#    # while not done, propagate labels
#    itercount = 0
#
#    # Show the initial labels
#    cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
#    pylab.imshow(host_labels)
#    pylab.title(itercount)
#    pylab.colorbar()
#    pylab.show()
#
##    cl.enqueue_copy(queue, gpu_done_flag, host_done_flag, is_blocking=False)
##    prop_exec = program.propagate_labels(queue, global_size, local_size,
##                                             gpu_labels, gpu_done_flag,
##                                             gpu_local_memory,
##                                             width, height,
##                                             buf_size[0], buf_size[1],
##                                             halo)
#
#    show_progress = True
#    total_time = 0
#
#    while True:
#        itercount += 1
#        host_done_flag[0] = 0
#        print 'iter', itercount
#        cl.enqueue_copy(queue, gpu_done_flag, host_done_flag, is_blocking=False)
#        prop_exec = program.propagate_labels(queue, global_size, local_size,
#                                             gpu_labels, gpu_done_flag,
#                                             gpu_local_memory,
#                                             width, height,
#                                             buf_size[0], buf_size[1],
#                                             halo)
#        prop_exec.wait()
#        elapsed = 1e-6 * (prop_exec.profile.end - prop_exec.profile.start)
#        total_time += elapsed
#        # read back done flag, block until it gets here
#        cl.enqueue_copy(queue, host_done_flag, gpu_done_flag, is_blocking=True)
#        if host_done_flag[0] == 0:
#            # no changes
#            break
#        # there were changes, so continue running
#        print host_done_flag
#        if itercount % 100 == 0 and show_progress:
#            cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
#            pylab.imshow(host_labels)
#            pylab.title(itercount)
#            pylab.show()
#        if itercount % 10000 == 0:
#            print 'Reached maximal number of iterations, aborting'
#            sys.exit(0)
#
#    print('Finished after {} iterations, {} ms total, {} ms per iteration'.format(itercount, total_time, total_time / itercount))
#    # Show final result
#    cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
#    print 'Found {} regions'.format(len(np.unique(host_labels)) - 1)
#    pylab.imshow(host_labels)
#    pylab.title(itercount)
#    pylab.show()
