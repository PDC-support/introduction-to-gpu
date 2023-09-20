#!/bin/bash
#SBATCH -A edu23.introgpu
#SBATCH -t 10
#SBATCH -N 1
#SBATCH -p gpu
###SBATCH -res=<Reservation name>

module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

module list

srun -n 1 ./hello
