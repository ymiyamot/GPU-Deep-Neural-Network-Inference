# Source File Description
### 1. python scripts

### a) run_single.py : run a single test
    * arguments : 
        - optim_type : optimization type applied
            - naive, blocked, vectorized, unrolled
        - optim_param : optimization parameters
            - Block size : [4, 8, 16, 32]
            - Vector size : [2, 4, 8, 16]
            - Unrolling factor : [2, 4, 8, 16]
        - network_sz : neural network size
            - small : 64(input) x 64 x 64 x 64 x 64(output)
            - medium :  64(input) x 256 x 256 x 256 x 64(output)
            - large :  64(input) x 1024 x 1024 x 1024 x 64(output)
        - n_inputs : the number of inputs
    * Returns
        - valid : check if parallel outputs match with serial results
        - runtime : GPU runtime
    * How to use
        python run_single.py blocked 4 large 1024
    ==> Run a signle test for blocked version with block size of 4 on large NNs with 1024 inputs.
    
### b) run_sweeps.py : Sweeps simulation
    * arguments : 
        - optim_type : same as run_single.py
        - network_sz : same as run_single.py
        - n_inputs : same as run_single.py
        - iters : the number of iterations to get average runtime
    * ex) python run_sweeps.py blocked large 1024 100
    ==> Do sweeps for blocked version on large NNs with 1024 inputs by changing block size [4, 8, 16, 32]
    * results : generate reports
        - In ./results/, report files are generated

---
### 2. Opencl kernels
### a) NN_naive.cl : GPU naive implementation
    - __kernel void NN_gpu_naive
### b) NN_blocked.cl : Blocked version of GPU implementation
    - __kernel void NN_gpu_blocked (Block size 4, 8, 16, 32)
### c) NN_vectortype.cl : Vectorized version of GPU implementation
    - __kernel void NN_gpu_vector2 : vector size 2
    - __kernel void NN_gpu_vector4 : vector size 4
    - __kernel void NN_gpu_vector8 : vector size 8
    - __kernel void NN_gpu_vector16 : vector size 16
 
### d) NN_vectortype_forrollout.cl : Unrolled version of GPU implementation
    - __kernel void NN_gpu_rollout2 : unrolling factor 2
    - __kernel void NN_gpu_rollout4 : unrolling factor 4
    - __kernel void NN_gpu_rollout8 : unrolling factor 8
    - __kernel void NN_gpu_rollout16 : unrolling factor 16
