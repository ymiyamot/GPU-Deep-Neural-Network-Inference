float4
sigmoid(float4 x)
{
    return(1 / (1 + exp(-x)));
}

//float4
//ReLU(float4 x)
//{
//    float ref = 0;
//    float4 res = (x > ref) & x;
//    return(res);
//}


__kernel void
NN_gpu_rollout1(__global float8 *inputs,
                __global __read_only float *weights,
                __global __write_only float8 *outputs,
                __local float8 *local_inputs,
                __local float *local_weights,
                int n_prev,
                int n_inputs,
                int n_next,
                int weight_begin)
{
    // Number of elements in vector type
    const int vector_type = 8;
    
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
    float8 acc0 = 0;
    
    
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
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // This is assuming that vector_type = 4.
        for (int elem_i = 0; elem_i < lszx * vector_type; elem_i++) {
            acc0 += local_weights[ly * lszx * vector_type + elem_i]
            * local_inputs[elem_i * lszx + lx];
        }
        
        // Synchronize the workers at this point, so that local memory
        // doesn't get refreshed to something new before workers are done
        // computing with it.
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    
    outputs[gx + n_inputs / vector_type * gy] = acc0;
    
}

__kernel void
NN_gpu_rollout2(__global float8 *inputs,
                __global __read_only float *weights,
                __global __write_only float8 *outputs,
                __local float8 *local_inputs,
                __local float *local_weights,
                int n_prev,
                int n_inputs,
                int n_next,
                int weight_begin)
{
    // Number of elements in vector type
    const int vector_type = 8;
    
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
    float8 acc0 = 0;
    float8 acc1 = 0;
    
    
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
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // This is assuming that vector_type = 4.
        for (int elem_i = 0; elem_i < lszx * 4; elem_i++) {
            acc0 += local_weights[ly * lszx * vector_type + 2 * elem_i]
            * local_inputs[(2 * elem_i) * lszx + lx];
            acc1 += local_weights[ly * lszx * vector_type + 2 * elem_i + 1]
            * local_inputs[(2 * elem_i + 1) * lszx + lx];
        }
        
        // Synchronize the workers at this point, so that local memory
        // doesn't get refreshed to something new before workers are done
        // computing with it.
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    
    outputs[gx + n_inputs / vector_type * gy] = acc0 + acc1;
    
}

__kernel void
NN_gpu_rollout4(__global float8 *inputs,
                __global __read_only float *weights,
                __global __write_only float8 *outputs,
                __local float8 *local_inputs,
                __local float *local_weights,
                int n_prev,
                int n_inputs,
                int n_next,
                int weight_begin)
{
    // Number of elements in vector type
    const int vector_type = 8;
    
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
    float8 acc0 = 0;
    float8 acc1 = 0;
    float8 acc2 = 0;
    float8 acc3 = 0;
    
    
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
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // This is assuming that vector_type = 4.
        for (int elem_i = 0; elem_i < lszx * 2; elem_i++) {
            acc0 += local_weights[ly * lszx * vector_type + 4 * elem_i]
            * local_inputs[(4 * elem_i) * lszx + lx];
            acc1 += local_weights[ly * lszx * vector_type + 4 * elem_i + 1]
            * local_inputs[(4 * elem_i + 1) * lszx + lx];
            acc2 += local_weights[ly * lszx * vector_type + 4 * elem_i + 2]
            * local_inputs[(4 * elem_i + 2) * lszx + lx];
            acc3 += local_weights[ly * lszx * vector_type + 4 * elem_i + 3]
            * local_inputs[(4 * elem_i + 3) * lszx + lx];
        }
        
        // Synchronize the workers at this point, so that local memory
        // doesn't get refreshed to something new before workers are done
        // computing with it.
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    
    outputs[gx + n_inputs / vector_type * gy] = acc0 + acc1 + acc2 + acc3;
    
}


__kernel void
NN_gpu_rollout8(__global float8 *inputs,
               __global __read_only float *weights,
               __global __write_only float8 *outputs,
               __local float8 *local_inputs,
               __local float *local_weights,
               int n_prev,
               int n_inputs,
               int n_next,
               int weight_begin)
{
    // Number of elements in vector type
    const int vector_type = 8;
    
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
    float8 acc0 = 0;
    float8 acc1 = 0;
    float8 acc2 = 0;
    float8 acc3 = 0;
    float8 acc4 = 0;
    float8 acc5 = 0;
    float8 acc6 = 0;
    float8 acc7 = 0;

    
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
        }
        barrier(CLK_LOCAL_MEM_FENCE);
  
        // This is assuming that vector_type = 4.
        for (int elem_i = 0; elem_i < lszx; elem_i++) {
            acc0 += local_weights[ly * lszx * vector_type + 8 * elem_i]
            * local_inputs[(8 * elem_i) * lszx + lx];
            acc1 += local_weights[ly * lszx * vector_type + 8 * elem_i + 1]
            * local_inputs[(8 * elem_i + 1) * lszx + lx];
            acc2 += local_weights[ly * lszx * vector_type + 8 * elem_i + 2]
            * local_inputs[(8 * elem_i + 2) * lszx + lx];
            acc3 += local_weights[ly * lszx * vector_type + 8 * elem_i + 3]
            * local_inputs[(8 * elem_i + 3) * lszx + lx];
            acc4 += local_weights[ly * lszx * vector_type + 8 * elem_i + 4]
            * local_inputs[(8 * elem_i + 4) * lszx + lx];
            acc5 += local_weights[ly * lszx * vector_type + 8 * elem_i + 5]
            * local_inputs[(8 * elem_i + 5) * lszx + lx];
            acc6 += local_weights[ly * lszx * vector_type + 8 * elem_i + 6]
            * local_inputs[(8 * elem_i + 6) * lszx + lx];
            acc7 += local_weights[ly * lszx * vector_type + 8 * elem_i + 7]
            * local_inputs[(8 * elem_i + 7) * lszx + lx];
        }
        
        // Synchronize the workers at this point, so that local memory
        // doesn't get refreshed to something new before workers are done
        // computing with it.
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    
    outputs[gx + n_inputs / vector_type * gy] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    
}


__kernel void
NN_gpu_rollout16(__global float8 *inputs,
                __global __read_only float *weights,
                __global __write_only float8 *outputs,
                __local float8 *local_inputs,
                __local float *local_weights,
                int n_prev,
                int n_inputs,
                int n_next,
                int weight_begin)
{
    // Number of elements in vector type
    const int vector_type = 8;
    
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
    float8 acc0 = 0;
    float8 acc1 = 0;
    float8 acc2 = 0;
    float8 acc3 = 0;
    float8 acc4 = 0;
    float8 acc5 = 0;
    float8 acc6 = 0;
    float8 acc7 = 0;
    float8 acc8 = 0;
    float8 acc9 = 0;
    float8 acc10 = 0;
    float8 acc11 = 0;
    float8 acc12 = 0;
    float8 acc13 = 0;
    float8 acc14 = 0;
    float8 acc15 = 0;
    
    
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
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // This is assuming that vector_type = 4.
        for (int elem_i = 0; elem_i < lszx / 2; elem_i++) {
            acc0 += local_weights[ly * lszx * vector_type + 16 * elem_i]
            * local_inputs[(16 * elem_i) * lszx + lx];
            acc1 += local_weights[ly * lszx * vector_type + 16 * elem_i + 1]
            * local_inputs[(16 * elem_i + 1) * lszx + lx];
            acc2 += local_weights[ly * lszx * vector_type + 16 * elem_i + 2]
            * local_inputs[(16 * elem_i + 2) * lszx + lx];
            acc3 += local_weights[ly * lszx * vector_type + 16 * elem_i + 3]
            * local_inputs[(16 * elem_i + 3) * lszx + lx];
            acc4 += local_weights[ly * lszx * vector_type + 16 * elem_i + 4]
            * local_inputs[(16 * elem_i + 4) * lszx + lx];
            acc5 += local_weights[ly * lszx * vector_type + 16 * elem_i + 5]
            * local_inputs[(16 * elem_i + 5) * lszx + lx];
            acc6 += local_weights[ly * lszx * vector_type + 16 * elem_i + 6]
            * local_inputs[(16 * elem_i + 6) * lszx + lx];
            acc7 += local_weights[ly * lszx * vector_type + 16 * elem_i + 7]
            * local_inputs[(16 * elem_i + 7) * lszx + lx];
            acc8 += local_weights[ly * lszx * vector_type + 16 * elem_i + 8]
            * local_inputs[(16 * elem_i + 8) * lszx + lx];
            acc9 += local_weights[ly * lszx * vector_type + 16 * elem_i + 9]
            * local_inputs[(16 * elem_i + 9) * lszx + lx];
            acc10 += local_weights[ly * lszx * vector_type + 16 * elem_i + 10]
            * local_inputs[(16 * elem_i + 10) * lszx + lx];
            acc11 += local_weights[ly * lszx * vector_type + 16 * elem_i + 11]
            * local_inputs[(16 * elem_i + 11) * lszx + lx];
            acc12 += local_weights[ly * lszx * vector_type + 16 * elem_i + 12]
            * local_inputs[(16 * elem_i + 12) * lszx + lx];
            acc13 += local_weights[ly * lszx * vector_type + 16 * elem_i + 13]
            * local_inputs[(16 * elem_i + 13) * lszx + lx];
            acc14 += local_weights[ly * lszx * vector_type + 16 * elem_i + 14]
            * local_inputs[(16 * elem_i + 14) * lszx + lx];
            acc15 += local_weights[ly * lszx * vector_type + 16 * elem_i + 15]
            * local_inputs[(16 * elem_i + 15) * lszx + lx];
        }
        
        // Synchronize the workers at this point, so that local memory
        // doesn't get refreshed to something new before workers are done
        // computing with it.
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    
    outputs[gx + n_inputs / vector_type * gy] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10 + acc11 + acc12 + acc13 + acc14 + acc15;
    
}




__kernel void
NN_gpu_vectortype_forrollout(__global float4 *inputs,
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
    float4 acc1 = 0;
    float4 acc2 = 0;
    float4 acc3 = 0;
    float4 acc4 = 0;

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

        // Original, before unrolling for loop
//        for (int elem_i = 0; elem_i < lszx * vector_type; elem_i++) {
//            acc += local_weights[ly * lszx * vector_type + elem_i] * local_inputs[elem_i * lszx + lx];
//        }
        
        // This is assuming that vector_type = 4.
        for (int elem_i = 0; elem_i < lszx; elem_i++) {
            acc1 += local_weights[ly * lszx * vector_type + 4 * elem_i]
            * local_inputs[(4 * elem_i) * lszx + lx];
            acc2 += local_weights[ly * lszx * vector_type + 4 * elem_i + 1]
            * local_inputs[(4 * elem_i + 1) * lszx + lx];
            acc3 += local_weights[ly * lszx * vector_type + 4 * elem_i + 2]
            * local_inputs[(4 * elem_i + 2) * lszx + lx];
            acc4 += local_weights[ly * lszx * vector_type + 4 * elem_i + 3]
            * local_inputs[(4 * elem_i + 3) * lszx + lx];
        }
        

        // Synchronize the workers at this point, so that local memory
        // doesn't get refreshed to something new before workers are done
        // computing with it.
        barrier(CLK_LOCAL_MEM_FENCE);

    }
    outputs[gx + n_inputs / vector_type * gy] = sigmoid(acc1 + acc2 + acc3 + acc4);

}
