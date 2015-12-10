# Analysis of GPU optimizations
(Hyunkwang Lee, Yohsuke Miyamoto)

### Blocked optimization (decomposing matrix multiplication into multiplications of sub-blocks)
In this blocked optimization, we divided up the weight matrix and input matrix into sub-matrices that were small enough to be stored in the local memory shared by each workgroup. In contrast, recall that the naive implementation performs the matrix multiplication by distributing to each worker one column of the inputs and one row of the weights .
We thought that there might be a tradeoff between 
We varied the size of the blocks from 1x1 to 32x32, square submatrices with row and column sizes being a power of 2.
More efficient use of local memory because there is more overlapping data used between workers within a sub-block of matrix rather than a long row by column multiplication.

<img src="./Plots/Perf_vs_blocksize.png"/>
Here we see that the 

### Vectorized optimization (using vectortypes)
In this vectorized optimization, we utilized vectortypes such as float 2, float4, float 8, float 16 to perform the neural network matrix multiplication. These vectortypes are variables that can store multiple 32-bit floats (e.g. float 8 stores 8 floats). Operations such as addition and multiplication act simultaneously on the multiple floats of a vector type variable. Thus, we thought we would be able to achieve considerable speed-up from using these vector types: ideally 4 times faster, for example, by using float4s.
Through this optimization, we get each worker to work on multiple outputs at once.
<img src="./Plots/Perf_vs_floattype.png"/>

### Unrolling optimization (For-loop unrolling)
In this unrolled optimization, we focused on the for loops that were being performed during the matrix multiplication.
By performing multiple operations within each iteration of the for-loop and correspondingly decreasing the number of iterations.
This has the benefit of decreasing the number of condition checks (checks performed after each iteration to determine whether to move to the next iteration).
We varied the number of operations being performed within each loop iteration from 1 (no change from previous) to 2, to 4, 8, 16.
<img src="./Plots/Perf_vs_unrolling.png"/>
