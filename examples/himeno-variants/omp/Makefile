# Note 1: We are explicitly linking to stdc++ library, because openmp seems to break 
#         this default linkage.
CCFLAGS = -std=c++14 -lstdc++ -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -O3

all:
	CC $(CCFLAGS) himeno_omp.cpp -DSSMALL -o himeno_ssmall
	CC $(CCFLAGS) himeno_omp.cpp -DSMALL -o himeno_small
	CC $(CCFLAGS) himeno_omp.cpp -DMIDDLE -o himeno_middle
	CC $(CCFLAGS) himeno_omp.cpp -DLARGE -o himeno_large
	CC $(CCFLAGS) himeno_omp.cpp -DELARGE -o himeno_elarge
