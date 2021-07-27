import numpy as np
import numba
import bisect

def random_draw(xvals, cdfvals, nsample=10):
    """
    randomly drawing from a discrete distribution
    """
    @numba.vectorize("int32(float64)")
    def invert_cdf(i):
        return bisect.bisect(cdfvals, i)-1
    x=np.random.rand(nsample)
    idx=invert_cdf(x)
    return np.array(xvals)[idx]


#job file for TSCC
#!/bin/bash
#PBS -q hotel
#PBS -N my_job
#PBS -l nodes=1:ppn=20
#PBS -l  walltime=0.05:00
#PBS -0 /oasis/tscc/scratch/path_to_dir
#PBS -e /oasis/tscc/scratch/path_to_dir
#PBS -V

qsub to submit [to submit]
qstat -u username [to check jobs]
