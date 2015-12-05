__kernel void
NN_gpu_vectortype(__global float4 *inputs,
             __global __read_only float4 *weights,
             __global __write_only float4 *outputs,
             __local float4 *local_inputs,
             __local float4 *local_weights,
             int n_prev,
             int n_inputs,
             int n_next,
             int weight_begin)
{
    // Global position of output pixel
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lszx = get_local_size(0);
    const int lszy = get_local_size(1);

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;
    
    // Register for accumulator
    float4 acc = 0;

    // Identifier for which workgroup we are in
    int wrkgrp_id_x = (gx / lszx);
    int wrkgrp_id_y = (gy / lszy);
    
    
    // Each thread is assigned multiplications for one row of weights,
    // corresponding to one output neuron

    
    // Determine which tile we're working on
    int total_tiles = n_prev / lszx;
    
    for (int curr_tile_i = 0; curr_tile_i < total_tiles; curr_tile_i++) {
        // iterate through tiles later with for loop
        if (idx_1D < lszx) {
            // Load tile of weights into local memory
            for (int row_i = 0; row_i < lszy; row_i++) {
                int x = idx_1D + curr_tile_i * lszx;
                int y = (wrkgrp_id_y * lszy + row_i);
                int ind = weight_begin + x + n_prev * y;
                local_weights[idx_1D + row_i * lszx] = weights[ind];
            }
            
            // Load tile of inputs into local memory
            for (int row_i = 0; row_i < lszy; row_i++) {
                int x = idx_1D + wrkgrp_id_x * lszx;
                int y = row_i + curr_tile_i * lszy;
                int ind = x + n_inputs * y;
                local_inputs[idx_1D + row_i * lszx] = inputs[ind];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int elem_i = 0; elem_i < lszx; elem_i++) {
            acc += local_weights[ly * lszx + elem_i] * local_inputs[elem_i * lszx + lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    outputs[gx + n_inputs * gy] = acc;

}

int
ReLU(int x)
{
    if (x < 0) {
        return 0;
    }else if (x >= 0) {
        return x;
    }
}

float
softmax(int x)
{
    if (x < 0) {
        return 0;
    }else if (x >= 0) {
        return x;
    }
}