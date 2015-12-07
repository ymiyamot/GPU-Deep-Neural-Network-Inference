
float4
sigmoid(float4 x)
{
    return(1 / (1 + exp(-x)));
}

__kernel void
NN_gpu_vectortype(__global float4 *inputs,
             __global __read_only float *weights,
             __global __write_only float4 *outputs,
             __local float4 *local_inputs,
             __local float *local_weights,
             int n_prev,
             int n_inputs,
             int n_next,
             int weight_begin)
{
    // Number of elements in vector type
    const int vector_type = 4;
    
    // Global position of output pixel
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lszx = get_local_size(0);
    const int lszy = get_local_size(1);


    // 1D index of thread within our work-group
    const int idx_1D = ly * lszx + lx;

    // Register for accumulator
    float4 acc = 0;

    // Identifier for which workgroup we are in
    int wrkgrp_id_x = gx / lszx;
    int wrkgrp_id_y = gy / lszy;

    
    // Determine which tile we're working on
    int total_tiles = n_prev / (lszx * vector_type);
    
    for (int curr_tile_i = 0; curr_tile_i < total_tiles; curr_tile_i++) {
        if (idx_1D < lszx * vector_type) {
            // Load tile of weights into local memory
            for (int row_i = 0; row_i < lszy; row_i++) {
                int x = idx_1D + curr_tile_i * lszx * vector_type;
                int y = (wrkgrp_id_y * lszy + row_i);
                int ind = weight_begin + x + n_prev * y;
                
                local_weights[idx_1D + row_i * lszx * vector_type] = weights[ind];
            }

            // Load tile of inputs into local memory
            for (int colm_i = 0; colm_i < lszx; colm_i++) {
                int x = (wrkgrp_id_x * lszx + colm_i);
                int y = idx_1D + curr_tile_i * lszy;
                int ind = x + (n_inputs / vector_type) * y;

                local_inputs[colm_i + idx_1D * lszx] = inputs[ind];
            }
            
//            // Print out all the local inputs
//            if (curr_tile_i == 0
//                & wrkgrp_id_x == 1
//                & wrkgrp_id_y == 0) {
//                for (int i = 0; i < 16; i++) {
//                    outputs[i] = local_inputs[i];
//                }
//            }
//            // Print out all the local weights
//            if (lx == 0 & ly == 0 & curr_tile_i == 0
//                & wrkgrp_id_x == 0
//                & wrkgrp_id_y == 0) {
//    
//                for (int i = 0; i < 16; i++) {
//                    outputs[i][0] = local_weights[(4 * i)];
//                    outputs[i][1] = local_weights[(4 * i) + 1];
//                    outputs[i][2] = local_weights[(4 * i) + 2];
//                    outputs[i][3] = local_weights[(4 * i) + 3];
//                }
//            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        for (int elem_i = 0; elem_i < lszx * vector_type; elem_i++) {
            acc += local_weights[ly * lszx * vector_type + elem_i] * local_inputs[elem_i * lszx + lx];
        }
        

        // Synchronize the workers at this point, so that local memory
        // doesn't get refreshed to something new before workers are done
        // computing with it.
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }

    outputs[gx + n_inputs / vector_type * gy] = sigmoid(acc);

}

