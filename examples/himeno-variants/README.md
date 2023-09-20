# Himeno Benchmark

The directory contains various implementation of the [himeno benchmark], the original one
that can be found on the benchmark's website can be found in the `original/` directory.
It serves as a base for porting to other parallelization frameworks.

*Note that* the original implementation is CPU single-threaded, and will ask for a matrix
size, you'll need to select 256x256x512, this is the implemented size in other 
implementations.

Each implementation's directory contains a make file, so you just need to `make` in the
directory to compile the program, **you also need to read the README if present**, for
example it's required to load some modules with SYCL on LUMI.

All implementations also provides a basic SLURM batch file, to run with 
`sbatch [options] job.slurm`.

Here are the different implementations (all are C++):
- `hip/`, an AMD HIP implementation with a simple-GPU computation with a stencil kernel,
  an attempt has been made to optimize the computation using shared memory, but is
  currently not worth it, so not enabled in the code. C++17 current, C++14 minimal.
  **Make sure** to load the environment show after this.
- `sycl/`, an hipSYCL (OpenSYCL) implementation, **make sure** to properly load the 
  modules, as explained in the directory's README.
- `omp/`, an OpenMP implementation, **make sure** to load the environment show after this.

Environment required for OpenMP and HIP to compile:
```
module load PrgEnv-aocc
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm
```

All benchmarks has been run on same LUMI hardware (excepting for single-threaded CPU 
reference code).

For matrice size 256x256x512:
- Original: ~1 GFLOPS
- HIP: ~178 GFLOPS
- SYCL: ~200 GFLOPS
- OpenMP: ~237 GFLOPS

We're sure that all of these implementations can be improved, in particular for memory
reads, because it seems to be the bottleneck.

[himeno benchmark]: https://i.riken.jp/en/supercom/documents/himenobmt/
