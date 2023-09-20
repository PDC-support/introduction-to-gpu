all:
	hipcc --amdgpu-target=gfx90a himeno_hip.cpp -std=c++17 -O3 -DSSMALL -o himeno_ssmall
	hipcc --amdgpu-target=gfx90a himeno_hip.cpp -std=c++17 -O3 -DSMALL -o himeno_small
	hipcc --amdgpu-target=gfx90a himeno_hip.cpp -std=c++17 -O3 -DMIDDLE -o himeno_middle
	hipcc --amdgpu-target=gfx90a himeno_hip.cpp -std=c++17 -O3 -DLARGE -o himeno_large
	hipcc --amdgpu-target=gfx90a himeno_hip.cpp -std=c++17 -O3 -DELARGE -o himeno_elarge
