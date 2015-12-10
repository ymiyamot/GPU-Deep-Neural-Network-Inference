from __future__ import division
import os
import sys
import pyopencl as cl
import numpy as np
import run_single
import NN_serial

if __name__ == '__main__':
    # Get arguments
    optim_type = str(sys.argv[1])
    network_sz = str(sys.argv[2])
    n_inputs = np.int32(sys.argv[3])
    iters = np.int32(sys.argv[4])
    cwd = os.getcwd()

    # Check available arguments
    if not network_sz in ["small", "medium", "large"]:
        raise Exception("For Neural Network Size, You need to use one among small, medium, large \n")
    if not optim_type in ["naive", "blocked", "vectorized", "unrolled"]:
        raise Exception("For optimization technique, You need to use one among naive, block, vector, unroll \n")

    # Set-ups
    if optim_type == "naive":
        pass
    elif optim_type == "blocked":
        sweep_confs = np.int32([4,8,16,32])
    elif optim_type == "vectorized":
        sweep_confs =  np.int32([2,4,8,16])
    elif optim_type == "unrolled":
        sweep_confs =  np.int32([2,4,8,16])

    runtimes_total = []
    valids_total = []
    # Do sweeps
    if optim_type == "naive":
        valid_naive, runtime_naive = run_single.main(optim_type, None, network_sz, n_inputs)
    else:
        for optim_param in sweep_confs:
            runtimes_single = []
            for iter in range(int(iters)):
                print("[iter:%s] Start test [%s-sized NN, %s optimization, %s size %s]"\
                   % (iter, str(network_sz), optim_type, optim_type, str(optim_param)))
                # Run a single test
                print(optim_type, optim_param, network_sz, n_inputs)
                valid, runtime = run_single.main(optim_type, optim_param, network_sz, n_inputs)
                runtimes_single.append(runtime)
                print("[%s iters Successfully Done! [%s-sized NN, %s optimization, %s size %s]" \
                   % (iters, network_sz, optim_type, optim_type, str(optim_param)))    

            valids_total.append(valid)
            runtimes_total.append(float(sum(runtimes_single))/float(len(runtimes_single)))


    # Write result reports
    result_dir = cwd+"/results/"+str(optim_type)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if network_sz == 'small':
        N_ops = 33554432
    elif network_sz == 'medium':
        N_ops = 335544320
    elif network_sz == 'large':
        N_ops = 4563401752

    if optim_type == "naive":
        summary_file = "%s_%s" % (optim_type,network_sz)
        summary = open(result_dir+"/"+summary_file, 'w')
        summary.write("=======================\n")
        summary.write("      GPU Results      \n")
        summary.write("=======================\n")
        summary.write("Optimization Technique   : %s\n" % str(optim_type))
        summary.write("Number of Inputs         : %s\n" % str(n_inputs))
        summary.write("Match with Serial?       : %s \n" % str(valid_naive))
        summary.write("Runtime                  : %s sec\n" % str(runtime_naive))
        summary.write("Performance              : %s GFLOPS\n" % str(N_ops/(runtime_naive*1e9)))
        summary.write("=======================\n")
        summary.write("      GPU Results      \n")
        summary.write("=======================\n")
    else:
        summary_file = "%s_%s" % (optim_type,network_sz)
        summary = open(result_dir+"/"+summary_file, 'w')
        summary.write("=======================\n")
        summary.write("      GPU Results      \n")
        summary.write("=======================\n")
        summary.write("Optimization Technique   : %s\n" % str(optim_type))
        summary.write("Number of Inputs         : %s\n" % str(n_inputs))
        for i in range(len(runtimes_total)):
            summary.write("---- %s Size : %s ----   : \n" % (str(optim_type), str(sweep_confs[i])))
            summary.write("Match with Serial?       : %s \n" % str(valids_total[i]))
            summary.write("Runtime                  : %s sec\n" % str(runtimes_total[i]))
            summary.write("Performance              : %s GFLOPS\n" % str(N_ops/(runtimes_total[i]*1e9)))
        summary.write("=======================\n")
        summary.write("      GPU Results      \n")
        summary.write("=======================\n")

    summary.close()
