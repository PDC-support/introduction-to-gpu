#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <cstddef>


using std::size_t;
using std::uint32_t;
using std::vector;
using std::cout;
using std::endl;
using std::flush;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;

using sycl::range;
using sycl::buffer;
using sycl::queue;
using sycl::handler;
using sycl::read_only;
using sycl::write_only;
using sycl::read_write;
using sycl::accessor;
using sycl::plus;
using sycl::id;
using sycl::property::queue::in_order;

#include "../paramset.h"

#define INDEX(it) ((it)[0] * COLS * DEPS + (it)[1] * DEPS + (it)[2])

size_t flop() {
    return (double)(COLS - 2) * (double)(ROWS - 2) * (double)(DEPS - 2) * 34;
}

double flops(double seconds, int nn) {
    return (double) flop() / seconds * (double) nn;
}

double jacobi(int nn, queue& q) {

    constexpr int cap = ROWS * COLS * DEPS;

    float *p = sycl::malloc_device<float>(cap, q);
    int *bnd = sycl::malloc_device<int>(cap, q);
    float *wrk1 = sycl::malloc_device<float>(cap, q);
    float *wrk2 = sycl::malloc_device<float>(cap, q);
    float *a[4] = {
        sycl::malloc_device<float>(cap, q),
        sycl::malloc_device<float>(cap, q),
        sycl::malloc_device<float>(cap, q),
        sycl::malloc_device<float>(cap, q),
    };
    float *b[3] = {
        sycl::malloc_device<float>(cap, q),
        sycl::malloc_device<float>(cap, q),
        sycl::malloc_device<float>(cap, q),
    };
    float *c[3] = {
        sycl::malloc_device<float>(cap, q),
        sycl::malloc_device<float>(cap, q),
        sycl::malloc_device<float>(cap, q),
    };
    float *gosa = sycl::malloc_shared<float>(1, q);

    q.submit([&](handler& cgh) {
        cgh.parallel_for(range<1>(cap), [=](id<1> it) {
            p[it] = 0.0;
            bnd[it] = 1;
            wrk1[it] = 0.0;
            wrk2[it] = 0.0;
            a[0][it] = 1.0;
            a[1][it] = 1.0;
            a[2][it] = 1.0;
            a[3][it] = 1.0 / 6.0;
            b[0][it] = 0.0;
            b[1][it] = 0.0;
            b[2][it] = 0.0;
            c[0][it] = 1.0;
            c[1][it] = 1.0;
            c[2][it] = 1.0;
        });
    });
    q.wait();

    q.submit([&](handler& cgh) {
        cgh.parallel_for(range<3>(ROWS, COLS, DEPS), [=](id<3> it) {
            p[INDEX(it)] = (float)(it[0] * it[0]) / (float)((ROWS - 1) * (ROWS - 1));
        });
    });
    q.wait();

    auto start = steady_clock::now();

    for (int n = 0; n < nn; n++) {
        
        *gosa = 0.0;

        auto gosa_sum = reduction(gosa, plus<float>());
        
        q.submit([&](handler& cgh) {
            cgh.parallel_for(range<3>(COLS - 2, ROWS - 2, DEPS - 2), gosa_sum, [=](id<3> it, auto& sum) {

                // Starting at 1 in buffers.
                it += id(1, 1, 1);

                int index = INDEX(it);

                // Compute s0...
                float s0 = a[0][index] * p[INDEX(it + id(1, 0, 0))]
                         + a[1][index] * p[INDEX(it + id(0, 1, 0))]
                         + a[2][index] * p[INDEX(it + id(0, 0, 1))]
                         + b[0][index] * ( p[INDEX(it + id(1, 1, 0))]  - p[INDEX(it + id(1, -1, 0))]
                                         - p[INDEX(it + id(-1, 1, 0))] + p[INDEX(it + id(-1, -1, 0))] )
                         + b[1][index] * ( p[INDEX(it + id(0, 1, 1))]  - p[INDEX(it + id(0, -1, 1))]
                                         - p[INDEX(it + id(0, 1, -1))] + p[INDEX(it + id(0, -1, -1))] )
                         + b[2][index] * ( p[INDEX(it + id(1, 0, 1))]  - p[INDEX(it + id(-1, 0, 1))]
                                         - p[INDEX(it + id(1, 0, -1))] + p[INDEX(it + id(-1, 0, -1))] )
                         + c[0][index] * p[INDEX(it + id(-1, 0, 0))]
                         + c[1][index] * p[INDEX(it + id(0, -1, 0))]
                         + c[2][index] * p[INDEX(it + id(0, 0, -1))]
                         + wrk1[index];

                float ss = (s0 * a[3][INDEX(it)] - p[INDEX(it)] ) * bnd[INDEX(it)];
                sum += ss * ss;

                wrk2[INDEX(it)] = p[INDEX(it)] + OMEGA * ss;

            });
        });
        q.wait();

        q.submit([&](handler& cgh) {
            cgh.parallel_for(range<3>(COLS - 2, ROWS - 2, DEPS - 2), [=](id<3> it) {
                it += id(1, 1, 1);
                p[INDEX(it)] = wrk2[INDEX(it)];
            });
        });
        q.wait();

    }

    auto stop = steady_clock::now();
    double time = duration_cast<duration<double>>(stop - start).count();
    cout << "  time: " << time << ", gosa: " << *gosa << ", mflops: " << flops(time, nn) / 1e6 << "\n";
    
    return time;

}

int main() {

    // queue q { sycl::cpu_selector(), in_order() };
    queue q { sycl::gpu_selector(), in_order() };

    cout << "Queue device:\n";
    cout << "  Device name: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    cout << "  Device vendor: " << q.get_device().get_info<sycl::info::device::vendor>() << "\n";
    cout << "  Device version: " << q.get_device().get_info<sycl::info::device::version>() << "\n";
    cout << "  Device driver version: " << q.get_device().get_info<sycl::info::device::driver_version>() << "\n";

    cout << "Himeno config:\n";
    cout << "  rows: " << ROWS << ", cols: " << COLS << ", deps: " << DEPS << "\n";
    cout << "  omega: " << OMEGA << "\n";
    
    float target = 60.0;
    
    int nn = 3;
    cout << "Start rehearsal measurement process (" << nn << " times)...\n" << flush;
    double time = jacobi(nn, q);
    nn = (int) (target / (time / 3.0));
    cout << "Start actual measurement process (" << nn << " times)...\n" << flush;
    jacobi(nn, q);

}
