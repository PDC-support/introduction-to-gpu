#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>

using std::cout;
using std::flush;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;


#include "../paramset.h"


using FloatMatrix = float[ROWS][COLS][DEPS];
using IntMatrix = int[ROWS][COLS][DEPS];

struct jacobi {
    FloatMatrix    p;
    IntMatrix    bnd;
    FloatMatrix wrk1;
    FloatMatrix wrk2;
    FloatMatrix a[4];
    FloatMatrix b[3];
    FloatMatrix c[3];
};


size_t flop() {
    return (double)(COLS - 2) * (double)(ROWS - 2) * (double)(DEPS - 2) * 34;
}

double flops(double seconds, int nn) {
    return (double) flop() / seconds * (double) nn;
}

double jacobi(int nn) {

    // Would be better to use unique_ptr smart pointer, but I don't know 
    // how to properly use it with OpenMP.
    struct jacobi *data = new struct jacobi;
    double time = 0.0;

    // Initialize matrices.
    // cout << "  initializing matrices...\n" << flush;
    // cout << "  starting...\n" << flush;

    #pragma omp target data map(from: data[0:1]) map(to: time)
    {

        // cout << "  initializing matrices...\n" << flush;
        
        #pragma omp target teams distribute parallel for collapse(3)
        for (int x = 0; x < ROWS; x++) {
            for (int y = 0; y < COLS; y++) {
                for (int z = 0; z < DEPS; z++) {
                    data->p[x][y][z] = (float) (x * x) / (float) ((ROWS - 1) * (ROWS - 1));
                    data->bnd[x][y][z] = 1;
                    data->wrk1[x][y][z] = 0.0;
                    data->wrk2[x][y][z] = 0.0;
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
            }
        }

        // cout << "  starting...\n" << flush;

        auto start = steady_clock::now();
        float gosa;

        for (int n = 0; n < nn; n++) {

            gosa = 0;
            // cout << "a...\n" << flush;

            #pragma omp target teams distribute parallel for collapse(3) reduction(+: gosa)
            for (int x = 1; x < ROWS - 1; ++x)
                for (int y = 1; y < COLS - 1; ++y)
                    for (int z = 1; z < DEPS - 1; ++z) {

                        float s0 
                            = data->a[0][x][y][z] * data->p[x+1][y  ][z  ]
                            + data->a[1][x][y][z] * data->p[x  ][y+1][z  ]
                            + data->a[2][x][y][z] * data->p[x  ][y  ][z+1]
                            + data->b[0][x][y][z] * ( data->p[x+1][y+1][z  ] - data->p[x+1][y-1][z  ]
                                                    - data->p[x-1][y+1][z  ] + data->p[x-1][y-1][z  ] )
                            + data->b[1][x][y][z] * ( data->p[x  ][y+1][z+1] - data->p[x  ][y-1][z+1]
                                                    - data->p[x  ][y+1][z-1] + data->p[x  ][y-1][z-1] )
                            + data->b[2][x][y][z] * ( data->p[x+1][y  ][z+1] - data->p[x-1][y  ][z+1]
                                                    - data->p[x+1][y  ][z-1] + data->p[x-1][y  ][z-1] )
                            + data->c[0][x][y][z] * data->p[x-1][y  ][z  ]
                            + data->c[1][x][y][z] * data->p[x  ][y-1][z  ]
                            + data->c[2][x][y][z] * data->p[x  ][y  ][z-1]
                            + data->wrk1[x][y][z];

                        float ss = (s0 * data->a[3][x][y][z] - data->p[x][y][z] ) * data->bnd[x][y][z];
                        gosa += ss * ss;

                        data->wrk2[x][y][z] = data->p[x][y][z] + OMEGA * ss;

                    }

            // cout << "b... " << gosa << "\n" << flush;
   
            #pragma omp target teams distribute parallel for collapse(3)
            for (int x = 1; x < ROWS - 1; ++x)
                for (int y = 1; y < COLS - 1; ++y)
                    for (int z = 1; z < DEPS - 1; ++z) {
                        data->p[x][y][z] = data->wrk2[x][y][z];
                    }
            
        }
        
        auto stop = steady_clock::now();
        time = duration_cast<duration<double>>(stop - start).count();
        cout << "  time: " << time << ", gosa: " << gosa << ", mflops: " << flops(time, nn) / 1e6 << "\n" << flush;

    }

    delete data;

    return time;

}


int main() {

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
