#!/bin/bash
#SBATCH -A edu23.introgpu
#SBATCH -t 10
#SBATCH -N 1
#SBATCH -p gpu
###SBATCH -res=<Reservation name>

module list

# time srun -n 1 rocprof --stats ./himeno
time srun -n 1 ./himeno
