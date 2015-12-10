# File Description
## python scripts

### run_single.py : run a single test
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
    
### run_sweeps.py : Sweeps simulation
    - arguments : 
        - optim_type : same as run_single.py
        - network_sz : same as run_single.py
        - n_inputs : same as run_single.py
        - iters : the number of iterations to get average runtime
    - ex) python run_sweeps.py block large 1024 100\
    ==> Do sweeps for **blocked** version on large NNs with 1024 inputs by changing block size [**2, 4, 8, 16**]
    - results : generate reports
        - In **./results/**, report files are generated
