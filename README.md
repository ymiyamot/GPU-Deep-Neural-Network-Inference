# CS205 (2015 Fall) Final Project
GPU-Deep-Neural-Network-Inference
(Hyunkwang Lee, Yohsuke Miyamoto)

## Project Aim
Run feed-forward DNN efficiently on GPU using multiple optimizations

## File Description
* NN_parallel.cl : Kernel codes for parallel implementation of DNN
* NN_serial.py : Serial implementation of DNN using numpy
* run_NNs.py : Initilization and launch kernels
* NN_naive.cl : Naive implementation of DNN (Our baseline)
* run_DNNs.py : Initilization and launch kernels on GPU (new version)
* run_DNNs.py : Initilization and launch kernels on GPU (new version)

## Optimizations
* Naive GPU implementation of DNN
   - Processed an input through the network at a time
   - Matrix-vector multiplication implementation

* Multiple threads within a workgroup

* Matrix Multiplication used to process a set of inputs at a time
   - Naive version of GPU Matrix multiplication

* Blocked Matrix Multiplication
   - reuse already-fetched data from global memory (memory spatial and temporal locality)
   - different block sizes

* Vectorized Matrix Multiplication
   - Intra-neuron multiplication
	- a vector of multiple elements are multiplied and only an output generated with a thread
   - Inter-neuron multilication 
	- 
 
* Loop Unrolling

* Bank conflicts

