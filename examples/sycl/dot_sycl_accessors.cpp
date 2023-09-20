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

  sycl::queue q{sycl::gpu_selector{}};

  // Initialize sum
  double sum = 0.0;
  // Use scope to ensure data is copied back to the host from the buffer
  {
    // Create a buffer for sum to get the reduction results
    sycl::buffer<double> bufSum{&sum, 1};
    // Create buffers for input vectors:
    sycl::buffer<double> bufA{vecA.data(), NX};
    sycl::buffer<double> bufB{vecB.data(), NX};

    // Submit a SYCL kernel into a queue
    q.submit([&](sycl::handler &cgh) {
      // Create accessors: read-write for the reduction variable, read-only for
      // inputs
      auto accSum = bufSum.get_access<sycl::access_mode::read_write>(cgh);
      auto accA = bufA.get_access<sycl::access_mode::read>(cgh);
      auto accB = bufB.get_access<sycl::access_mode::read>(cgh);
      // Create temporary object describing variables with reduction semantics
      // We can use built-in reduction primitive
      auto reduction = sycl::reduction(accSum, sycl::plus<double>());

      // A reference to the reducer is passed to the lambda
      cgh.parallel_for(sycl::range<1>{NX}, reduction,
                       [=](sycl::id<1> idx, auto &reducer) {
                         // Compute the pairwise product and do the reduction
                         reducer.combine(accA[idx] * accB[idx]);
                       });
    });
    // The contents of bufSum are copied back to sum by the destructor of bufSum
  }
  // Print results
  printf("sum = %lf\n", sum);
}
