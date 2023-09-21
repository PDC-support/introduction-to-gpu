#include <iostream>
#include <math.h>
#include <sycl/sycl.hpp>
#include <vector>

const int NX = 102400;

int main() {
  std::vector<double> vecA(NX), vecB(NX);
  double r = 0.2;

  // Initialization of vectors
  for (int i = 0; i < NX; i++) {
    vecA[i] = pow(r, i);
    vecB[i] = 1.0;
  }

  // Create an in-order queue
  sycl::queue q{sycl::gpu_selector{}, {sycl::property::queue::in_order()}};

  // Allocate arrays on the GPU
  double *vecA_d = sycl::malloc_device<double>(NX, q);
  double *vecB_d = sycl::malloc_device<double>(NX, q);
  double *sum_d = sycl::malloc_device<double>(1, q);

  // Fill the arrays on the GPU
  q.copy<double>(vecA.data(), vecA_d, NX);
  q.copy<double>(vecB.data(), vecB_d, NX);
  q.fill<double>(sum_d, 0.0, 1);

  // Submit a SYCL kernel into a queue
  q.submit([&](sycl::handler &cgh) {
    // Use SYCL redution functionality to combine computation and redution
    auto reduction = sycl::reduction(sum_d, sycl::plus<double>());

    // A reference to the reducer is passed to the lambda
    cgh.parallel_for(sycl::range<1>{NX}, reduction,
                     [=](sycl::id<1> idx, auto &reducer) {
                       // Compute the pairwise product and do the reduction
                       reducer.combine(vecA_d[idx] * vecB_d[idx]);
                     });
  });

  // Copy a single value to host, and wait for the copy to complete!
  double sum;
  q.copy<double>(sum_d, &sum, 1).wait();

  // Free the memory
  sycl::free(vecA_d, q);
  sycl::free(vecB_d, q);
  sycl::free(sum_d, q);

  // Print results
  printf("sum = %lf\n", sum);
}
