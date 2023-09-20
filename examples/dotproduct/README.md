# Build with

module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

hipcc --amdgpu-target=gfx90a solution_hip.hip -std=c++17 -O3 -o solution_hip
