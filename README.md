# CS205 (2015 Fall) Final Project
GPU-Deep-Neural-Network-Inference
(Hyunkwang Lee, Yohsuke Miyamoto)

## Project Aim
Run feed-forward DNN efficiently on GPU using multiple optimizations

## File Description
* NN_parallel.cl : Kernel codes for parallel implementation of DNN
* NN_serial.py : Serial implementation of DNN using numpy
* run_NNs.py : Initilization and launch kernels

## Optimizations
* Naive GPU implementation of DNN
   - Processed an input through the network at a time
   - Matrix-vector multiplication implementation

* Multiple threads within a workgroup

* Matrix Multiplication used to process a set of inputs at a time

* Blocked Matrix Multiplication
   - efficient usage of local memories (avoid bank conflicts)
   - 
