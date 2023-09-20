#include <rocprim/rocprim.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <chrono>

using std::cout;
using std::endl;
using std::flush;
using std::runtime_error;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;


#include "../paramset.h"


// Number of threads in the Z axis. X/Y are always one thread wide.
#define THREADS 64


using FloatMatrix = float[ROWS][COLS][DEPS];
using IntMatrix = int[ROWS][COLS][DEPS];

struct jacobi {
    FloatMatrix    p;
    IntMatrix    bnd;
    FloatMatrix wrk1;
    FloatMatrix wrk2;
    FloatMatrix   ss; // The HIP implementation uses an additional 'ss' matrix to be later summed into gosa.
    FloatMatrix a[4];
    FloatMatrix b[3];
    FloatMatrix c[3];
};


__global__ void jacobi_init(struct jacobi *data) {

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z * THREADS + threadIdx.x;

    data->p[x][y][z] = (float) (x * x) / (float) ((ROWS - 1) * (ROWS - 1));
    data->bnd[x][y][z] = 1;
    data->wrk1[x][y][z] = 0.0;
    data->wrk2[x][y][z] = 0.0;
    data->ss[x][y][z] = 0.0;
    data->a[0][x][y][z] = 1.0;
    data->a[1][x][y][z] = 1.0;
    data->a[2][x][y][z] = 1.0;
    data->a[3][x][y][z] = 1.0 / 6.0;
    data->b[0][x][y][z] = 0.0;
    data->b[1][x][y][z] = 0.0;
    data->b[2][x][y][z] = 0.0;
    data->c[0][x][y][z] = 1.0;
    data->c[1][x][y][z] = 1.0;
    data->c[2][x][y][z] = 1.0;

}

__global__ void jacobi_run_v0(struct jacobi *data) {

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z * THREADS + threadIdx.x;

    if (x == 0 || y == 0 || z == 0 || x == ROWS - 1 || y == COLS - 1 || z == DEPS - 1)
        return;

    // Compute s0...
    float s0 = data->a[0][x][y][z] * data->p[x + 1][y    ][z    ]
             + data->a[1][x][y][z] * data->p[x    ][y + 1][z    ]
             + data->a[2][x][y][z] * data->p[x    ][y    ][z + 1]
             + data->b[0][x][y][z] * ( data->p[x + 1][y + 1][z    ] - data->p[x + 1][y - 1][z    ]
                                     - data->p[x - 1][y + 1][z    ] + data->p[x - 1][y - 1][z    ] )
             + data->b[1][x][y][z] * ( data->p[x    ][y + 1][z + 1] - data->p[x    ][y + 1][z + 1]
                                     - data->p[x    ][y + 1][z - 1] + data->p[x    ][y - 1][z - 1] )
             + data->b[2][x][y][z] * ( data->p[x + 1][y    ][z + 1] - data->p[x - 1][y    ][z + 1]
                                     - data->p[x + 1][y    ][z - 1] + data->p[x - 1][y    ][z - 1] )
             + data->c[0][x][y][z] * data->p[x - 1][y    ][z    ]
             + data->c[1][x][y][z] * data->p[x    ][y - 1][z    ]
             + data->c[2][x][y][z] * data->p[x    ][y    ][z - 1]
             + data->wrk1[x][y][z];

    float ss = (s0 * data->a[3][x][y][z] - data->p[x][y][z] ) * data->bnd[x][y][z];

    data->ss[x][y][z] = ss * ss;
    data->wrk2[x][y][z] = data->p[x][y][z] + OMEGA * ss;

}

__global__ void jacobi_transfer(struct jacobi *data) {

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z * THREADS + threadIdx.x;
    
    if (x == 0 || y == 0 || z == 0 || x == ROWS - 1 || y == COLS - 1 || z == DEPS - 1)
        return;

    data->p[x][y][z] = data->wrk2[x][y][z];

}


void hip_check(hipError_t err) {
    if (err != hipSuccess) {
        std::ostringstream stream;
        stream << "hip error: " << err << " (" << hipGetErrorString(err) << ")";
        throw runtime_error(stream.str());
    }
}

size_t flop() {
    return (double)(COLS - 2) * (double)(ROWS - 2) * (double)(DEPS - 2) * 34;
}

double flops(double seconds, int nn) {
    return (double) flop() / seconds * (double) nn;
}

double jacobi(int nn) {

    struct jacobi *data;
    hip_check(hipMalloc(&data, sizeof(struct jacobi)));

    float gosa_h; // Host-alloc gosa
    float *gosa;  // Device-alloc gosa
    hip_check(hipMalloc(&gosa, sizeof(float)));

    void *tmp;
    hip_check(hipMalloc(&tmp, 32776 * 2));

    dim3 threads(THREADS);
    dim3 blocks(ROWS, COLS, DEPS / THREADS);

    hipLaunchKernelGGL(jacobi_init, blocks, threads, 0, 0, data);
    hip_check(hipGetLastError());
    hip_check(hipDeviceSynchronize());
    
    auto start = steady_clock::now();

    for (int n = 0; n < nn; n++) {

        hipLaunchKernelGGL(jacobi_run_v0, blocks, threads, 0, 0, data);
        // hipLaunchKernelGGL(jacobi_run_v1, blocks, dim3(THREADS_X + 2, THREADS_Y + 2, THREADS_Z + 2), 0, 0, data);
        // hipLaunchKernelGGL(jacobi_run_v1, blocks, threads, 0, 0, data);
        hip_check(hipGetLastError());
        hip_check(hipDeviceSynchronize());

        // Sum all 'ss' values to gosa
        size_t temporary_storage_size_bytes;
        auto ss = &data->ss[0][0][0];
        hip_check(rocprim::reduce(tmp, temporary_storage_size_bytes, ss, gosa, ROWS * COLS * DEPS));

        // Redefine 'p' from 'ss' values, omega and previous p value.
        hipLaunchKernelGGL(jacobi_transfer, blocks, threads, 0, 0, data);
        hip_check(hipDeviceSynchronize());

        hip_check(hipMemcpy(&gosa_h, gosa, sizeof(float), hipMemcpyDeviceToHost));
        hip_check(hipDeviceSynchronize());

    }

    auto stop = steady_clock::now();
    double time = duration_cast<duration<double>>(stop - start).count();
    cout << "  time: " << time << ", gosa: " << gosa_h << ", mflops: " << flops(time, nn) / 1e6 << "\n" << flush;

    hip_check(hipFree(data));
    hip_check(hipFree(gosa));
    hip_check(hipFree(tmp));

    return time;

}


int main() {

    int device_count;
    hip_check(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
        cout << "No device, aborting..." << "\n";
        return 1;
    }

    hipDeviceProp_t device_prop;
    hip_check(hipGetDeviceProperties(&device_prop, 0));

    cout << "Device:\n";
    cout << "  Name: " << device_prop.name << "\n";
    cout << "  GCN arch: " << device_prop.gcnArchName << "\n";
    cout << "  Total global memory: " << device_prop.totalGlobalMem << "\n";

    cout << "Himeno config:\n";
    cout << "  Rows: " << ROWS << ", Cols: " << COLS << ", Deps: " << DEPS << "\n";
    cout << "  Omega: " << OMEGA << "\n" << flush;

    float target = 60.0;
    
    int nn = 3;
    cout << "Start rehearsal measurement process (" << nn << " times)...\n" << flush;
    double time = jacobi(nn);
    nn = (int) (target / (time / 3.0));
    cout << "Start actual measurement process (" << nn << " times)...\n" << flush;
    jacobi(nn);

}
