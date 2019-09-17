#include<cuda.h>
#include<iostream>
#include<stdio.h>
#include<cuda.h>
#include "solution.h"



__global__ void kernel(float* a, float* b, float* c, int rows, int cols) {
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int i=0;
	float temp = 0;
	for(i =0; i<cols; i++){
		temp += a[i]*b[i*rows+tid];
	}
	c[tid] = temp;
	
}