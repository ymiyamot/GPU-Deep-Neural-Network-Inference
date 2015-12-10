# Analysis of GPU optimizations

## Methodology

We implemented 5 versions of the DNN: a non-GPU version in numpy, a naive GPU implementation, and 3 GPU implementations optimized using different techniques (Blocking, Vectorization, Unrolling, described below). However, the performance of each of these 3 optimizations can depend on parameters that depend on various factors such as GPU specifications, number of inputs, and network architecture. Thus we compared the performance speed across different values of these parameter configurations in order to determine what configurations performed best.

<img src="../Plots/Methodology.png"/>

### Correctness
While implementing additional optimizations, we verified correctness of our implementations, by comparing the output of our GPU implementations with the output from a non-GPU serial version of the DNN implemented using python's numpy package.

### Structure of test network
Our test networks consisted of 5 layers: 1 input layer, 1 output layer, and 3 hidden layers in between. In order to assess the generality of our test results across different amounts of computation, we implemented 3 different size networks, small, medium, large. To do this we varied the size of each of the 3 hidden layers from 64, 256, to 1024, while fixing the size of the input and output layers to be 64. In addition, we fixed the number of inputs to be computed to be 1024. The values of the weights and inputs were drawn randomly from a normal distribution.

### Baseline speed comparison
As a baseline comparison, we implemented a very basic version of the DNN on the GPU. In this version, each worker computes a single output by computing a vector dot product (multiplying an entire row of weights by an entire input vector). This implementation did not utilize efficient use of local memory, and each worker read its values from global memory.


---
### 1. Blocked optimization (decomposing matrix multiplication into multiplications of sub-blocks)
In this blocked optimization, we divided up the weight matrix and input matrix into sub-matrices that were small enough to be stored in the local memory shared by each workgroup. In contrast, recall that the naive implementation performs the matrix multiplication by distributing to each worker one column of the inputs and one row of the weights .
We thought that there might be a tradeoff between 
We varied the size of the blocks from 1x1 to 32x32, square submatrices with row and column sizes being a power of 2.
More efficient use of local memory because there is more overlapping data used between workers within a sub-block of matrix rather than a long row by column multiplication.

<img src="../Plots/Perf_vs_blocksize.png"/>
Here we see that the 
---
### 2. Vectorized optimization (using vectortypes)
In this vectorized optimization, we utilized vectortypes such as float 2, float4, float 8, float 16 to perform the neural network matrix multiplication. These vectortypes are variables that can store multiple 32-bit floats (e.g. float 8 stores 8 floats). Operations such as addition and multiplication act simultaneously on the multiple floats of a vector type variable. Thus, we thought we would be able to achieve considerable speed-up from using these vector types: ideally 4 times faster, for example, by using float4s.
Through this optimization, we get each worker to work on multiple outputs at once.
<img src="../Plots/Perf_vs_floattype.png"/>
---
### 3. Unrolling optimization (For-loop unrolling)
In this unrolled optimization, we focused on the for loops that were being performed during the matrix multiplication.
By performing multiple operations within each iteration of the for-loop and correspondingly decreasing the number of iterations.
This has the benefit of decreasing the number of condition checks (checks performed after each iteration to determine whether to move to the next iteration).
We varied the number of operations being performed within each loop iteration from 1 (no change from previous) to 2, to 4, 8, 16.
<img src="../Plots/Perf_vs_unrolling.png"/>
---
### 4. Summary of Results
<img src="../Plots/Perf_progression.png"/>
