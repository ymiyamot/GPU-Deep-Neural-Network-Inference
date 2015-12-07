float
sigmoid(float x)
{
    return(1 / (1 + exp(-x)));
}

__kernel void
NN_gpu_naive(__global float *neurons,
             __global __read_only float *weights,
             __local float *summed_val,
             int n_neurons_prev,
             int weight_begin)
{
    // Global position of output pixel
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;
    
    // Each thread is assigned multiplications for one row of weights,
    // corresponding to one output neuron
    summed_val[lx] += neurons[lx] * weights[weight_begin + n_neurons_prev * gy + lx];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    // Sum the result of the multiplications from previous step.
    if (lx == 0) {
        float total = 0;
        for (int val_i = 0; val_i < n_neurons_prev; val_i++) {
            total += summed_val[val_i];
        }
//        neurons[gy] = ReLU(total);
        neurons[gy] = sigmoid(total);
    }
}
