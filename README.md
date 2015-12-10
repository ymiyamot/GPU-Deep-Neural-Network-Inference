# CS205 (2015 Fall) Final Project
GPU-Deep-Neural-Network-Inference
(Hyunkwang Lee, Yohsuke Miyamoto)

## Project Aim
The purpose of this project was to implement DNN prediction on the GPU using openCL, and perform several optimizations to speed up this operation. 

## Background and motivation
Deep neural networks (DNNs) have resurged in popularity in the past decades, and are consistently reported to yield cutting-edge performance in many fields of machine learning, such as image classification, voice recognition, natural language processing, and more.  Since neural network computations (essentially large matrix multiplications) are well-suited for parallelization, researchers often use GPUs, FPGAs, or clusters to speed up network training and prediction. Although there exist many high-performance cutting-edge software packages already implemented (such as cuBLAS to perform matrix multiplications), we chose to implement this process from scratch to understand for ourselves what aspects of parallelism can be exploited and where the bottlenecks of performance lie in neural networks.
Neural networks typically consist of 2 stages: (1) training on example data to tune their weights (learning) and (2) prediction using the trained configuration on new data (inference). For the scope of this project we choose to focus simply on step 2, the prediction step, in which we propagate the inputs through the neural network, given a set of configured weights. We focused on this component of DNN usage because DNN prediction can benefit greatly due to it being a recurring cost (in contrast to the one-time training of the network) and fast real-time predictions can be important in many applications. Furthermore, although many complex structures of DNNs exist (e.g. convolutional neural networks, recurrent neural networks) for simplicity, we focused on the simpler case of fully connected feed-forward networks.


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
