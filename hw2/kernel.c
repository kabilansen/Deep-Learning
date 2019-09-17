// #include<iostream>
#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include "solution.h"



void kernel(float* a, float* b, float* c, int rows, int cols, int jobs) {
	// int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = omp_get_thread_num();
	//  printf("in Kernel 1r");
	int i, j, stop;
	float temp = 0;

	if((tid+1)*jobs > rows) stop=cols;
    else stop = (tid+1)*jobs;
	for(i = tid*jobs; i < stop; i++){
		temp = 0;
		for(j=0;j<cols;j++){
			temp += a[j]*b[j*rows+i];
		}
		
		// printf("%f\n",b[i+tid]);
		c[i] = temp;
	}
	// c[tid] = temp;

	// printf("%f",c[tid]);
	
}