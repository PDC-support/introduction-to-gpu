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

echo '--- Run himeno_ssmall ---'
time srun -n 1 ./himeno_ssmall

echo '--- Run himeno_small ---'
time srun -n 1 ./himeno_small

echo '--- Run himeno_middle ---'
time srun -n 1 ./himeno_middle

echo '--- Run himeno_large ---'
time srun -n 1 ./himeno_large

echo '--- Run himeno_elarge ---'
time srun -n 1 ./himeno_elarge
