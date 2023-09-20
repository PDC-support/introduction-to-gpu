/* Copyright (c) 2019 CSC Training */
/* Copyright (c) 2021 ENCCS */
#include <stdio.h>
#include <math.h>
#define NX 102400

int main(void)
{
  double *vecA,*vecB,*vecC;
  double r=0.2;

  vecA = (double*) malloc(NX*sizeof(double));
  vecB = (double*) malloc(NX*sizeof(double));
  vecC = (double*) malloc(NX*sizeof(double));			  
			  
/* Initialization of vectors */
  for (int i = 0; i < NX; i++) {
     vecA[i] = pow(r, i);
     vecB[i] = 1.0;
  }

/* dot product of two vectors */
  for (int i = 0; i < NX; i++) {
     vecC[i] = vecA[i] * vecB[i];
  }

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

