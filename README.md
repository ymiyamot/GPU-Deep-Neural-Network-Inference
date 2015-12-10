# CS205 (2015 Fall) Final Project
GPU-Deep-Neural-Network-Inference
(Hyunkwang Lee, Yohsuke Miyamoto)

## Project Aim
Run feed-forward DNN prediction efficiently on the GPU using multiple optimizations

## File Description
### python scripts

- run_single.py : run a single test
    - arguments : 
        - optim_type : optimization type applied
            - **naive, block, vector, unroll**
        - optim_param : optimization parameters
            - Block size : [**2, 4, 8, 16**]
            - Vector size : [**2, 4, 8, 16**]
            - Unrolling factor : [**2, 4, 8, 16**]
        - network_sz : neural network size
            - **small** : 64 x 64 x 64 x 64 x 64
            - **medium** :  64 x 256 x 256 x 256 x 64
            - **large** :  64 x 1024 x 1024 x 1024 x 64
        - n_inputs : the number of inputs
    - Returns
        - valid : check if parallel outputs match with serial results
        - runtime : GPU runtime
    - How to use\
        **python run_single.py block 4 large 1024**
    ==> Run a signle test for **block** version with block size of **4** on **large** NNs with **1024** inputs.
    
- run_sweeps.py : Sweeps simulation
    - arguments : 
        - optim_type : same as run_single.py
        - network_sz : same as run_single.py
        - n_inputs : same as run_single.py
        - iters : the number of iterations to get average runtime
    - ex) python run_sweeps.py block large 1024 100\
    ==> Do sweeps for **blocked** version on large NNs with 1024 inputs by changing block size [**2, 4, 8, 16**]
    - results : generate reports
        - In **./results/**, report files are generated

### Opencl kernels
* NN_naive.cl : GPU naive implementation
    - __kernel void NN_gpu_naive
* NN_blocked.cl : Blocked version of GPU implementation
    - __kernel void NN_gpu_blocked
* NN_vectortype.cl : Vectorized version of GPU implementation
    - __kernel void NN_gpu_vector2 : vector size 2
    - __kernel void NN_gpu_vector4 : vector size 4
    - __kernel void NN_gpu_vector8 : vector size 8
    - __kernel void NN_gpu_vector16 : vector size 16
 
* NN_vectortype_forrollout.cl : Unrolled version of GPU implementation
    - __kernel void NN_gpu_rollout1 : unrolling factor 1 (unrolling not applied)
    - __kernel void NN_gpu_rollout2 : unrolling factor 2
    - __kernel void NN_gpu_rollout4 : unrolling factor 4
    - __kernel void NN_gpu_rollout8 : unrolling factor 8
    - __kernel void NN_gpu_rollout16 : unrolling factor 16

## Analysis
- [Analysis Writeups](https://github.com/ymiyamot/GPU-Deep-Neural-Network-Inference/tree/master/ANALYSIS.md)
