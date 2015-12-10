from __future__ import division
import sys
import pyopencl as cl
import numpy as np
#import run_single
import NN_serial

if __name__ == '__main__':
    # Get arguments
    NN_type = sys.argv[1]
    opt = sys.argv[2]
    N_input = sys.argv[3]
    iterations = sys.argv[4]
    cwd = os.getcwd()

    # Check available arguments
    if not NN_type in ["small", "medium", "large"]:
        raise Exception("For Neural Network Size, You need to use one among small, medium, large \n")
    if not opt in ["naive", "block", "vector", "unroll"]:
        raise Exception("For optimization technique, You need to use one among naive, block, vector, unroll \n")
    if not iterations.isdigit():
        raise Exception("For iterations, You need to use integer numbers \n")

    # Set-ups
    if opt == "naive":
        pass
    elif opt == "block":
        sweep_confs = [2,4,8,16,32]
    elif opt == "vector":
        sweep_confs = [2,4,8,16]
    elif opt == "unroll":
        sweep_confs = [2,4,8,16]

    runtimes_total = []
    valids_total = []
    # Do sweeps
    if opt == "naive":
        valid, runtime = run_single.main(NN_type,opt,conf,N_input)
    else:
        for conf in sweep_confs:
            runtimes_single = []
            for iter in range(int(iterations)):
                print("[iter:%s] Run test [%s-sized NN, %s optimization, %s size %s]"\
                   % (iter, NN_type, opt, opt, str(conf)))    
                # Run a single test
                valid, runtime = run_single.main(NN_type,opt,conf,N_input)
                runtimes_single.append(runtime)
                print("[%s iterations Successfully Done! [%s-sized NN, %s optimization, %s size %s]" \
                   % (iterations, NN_type, opt, opt, str(conf)))    

            valids_total.append(valid)
            runtimes_total.append(float(sum(runtimes_single))/float(len(runtimes_single)))


    # Write result reports
    result_dir = cwd+"/results/"+str(opt)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    summary_file = "%s_%s" % (opt,NN_type)
    summary = open(result_dir+"/"+summary_file, 'w')
    summary.write("=======================\n")
    summary.write("      GPU Results      \n")
    summary.write("=======================\n")
    summary.write("Optimization Technique   : %s\n" % str(opt))
    summary.write("Number of Inputs         : %s\n" % str(N_input))

    for i in range(len(runtimes_total)):
        summary.write("---- %s Size : %s ----   : \n" % (str(opt), str(sweep_confs[i])))
        summary.write("Match with Serial?       : %s \n" % (str(valid_total[i]))
        summary.write("Runtime                  : %s sec\n" % str(runtimes_total[i]))
        summary.write("Performance              : %s GFLOPS\n" % str(N_ops/(runtimes_total[i]*1e9)))

    summary.write("=======================\n")
    summary.write("      GPU Results      \n")
    summary.write("=======================\n")

    write.close()
