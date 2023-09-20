/* Copyright (c) 2019 CSC Training */
/* Copyright (c) 2021 ENCCS */
#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>

#define NX 102400

#define WORKGROUP_SIZE 1024

__global__
void dot_product(double *vecC, double *vecA, double *vecB)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i<NX) {
     vecC[i] = vecA[i] * vecB[i];
  }
}

int main(void)
{
  double *vecA,*vecB,*vecC;
  double *vecA_d,*vecB_d,*vecC_d;
  double r=0.2;

  vecA = (double*) malloc(NX*sizeof(double));
  vecB = (double*) malloc(NX*sizeof(double));
  vecC = (double*) malloc(NX*sizeof(double));			  

/* allocate device memory */  
  hipMalloc((void**)&vecA_d,NX*sizeof(double) );
  hipMalloc((void**)&vecB_d,NX*sizeof(double) );
  hipMalloc((void**)&vecC_d,NX*sizeof(double) );
  
/* Initialization of vectors */
  for (int i = 0; i < NX; i++) {
     vecA[i] = pow(r, i);
     vecB[i] = 1.0;
  }

/* copy to device */  
  hipMemcpy(vecA_d,vecA,NX*sizeof(double),hipMemcpyHostToDevice);
  hipMemcpy(vecB_d,vecB,NX*sizeof(double),hipMemcpyHostToDevice);
  
  int workgroups = (NX + WORKGROUP_SIZE - 1)/WORKGROUP_SIZE;
  hipLaunchKernelGGL(dot_product,dim3(workgroups),dim3(WORKGROUP_SIZE),0,0,
                     vecC_d,vecA_d,vecB_d);
  
/* copy result back */
  hipMemcpy(vecC,vecC_d,NX*sizeof(double),hipMemcpyDeviceToHost);
  
  double sum = 0.0;
  /* calculate the sum */
  for (int i = 0; i < NX; i++) {
    sum += vecC[i];
  }
  printf("The sum is: %8.6f \n", sum);

  free(vecA);
  free(vecB);
  free(vecC);
  
  return 0;
}
