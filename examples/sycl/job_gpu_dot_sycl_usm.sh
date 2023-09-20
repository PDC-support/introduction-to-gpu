#!/bin/bash
#SBATCH -A <TODO> # Set the allocation to be charged for this job
#SBATCH -J myjob # Name of the job
#SBATCH -p gpu # The partition
#SBATCH -t 00:05:00 # 5 minutes wall-clock time
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of MPI processes per node

ml PDC/22.06
ml hipsycl/0.9.4-cpeGNU-22.06-rocm-5.3.3 

srun ./dot_sycl_usm > output.txt # Run the executable
