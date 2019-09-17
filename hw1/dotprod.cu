#include<cuda.h>
#include<iostream>
#include<stdio.h>
#include<cuda.h>
#include "solution.h"

#define imin(a,b) (a<b?a:b)

const int numberofThreads = 3;
const int threadsPerBlock = 3;
// const int blocksPerGrid = imin(32, (numberofThreads+threadsPerBlock-1) / threadsPerBlock);




int main (int argc, char* argv[]) {

	//dynamic variables
	int rows, cols, threads, CUDA_DEVICE;
	

	rows = atoi(argv[1]);
	cols = atoi(argv[2]);
	CUDA_DEVICE = atoi(argv[5]);
	threads = atoi(argv[6]);

	// printf("%d %d", rows, cols);
    FILE *fp, *fv;
	size_t size;
	size_t vec_Size;
	size = (size_t)((size_t)rows * (size_t)cols);
	vec_Size = (size_t)((size_t)1*(size_t)cols);
	float c;
	float* partial_c = (float*) malloc(rows*sizeof(float));
	float *w_vect=(float*)malloc((vec_Size)*sizeof(float));
	float *b = (float*)malloc((size)*sizeof(float));
    float *dev_a, *dev_b, *dev_partial_c;
    char *line = NULL; size_t len = 0;
	char *token, *saveptr;
	float file_data;
	float mat[rows][cols];
	
	
	// partial_c = (float*)malloc((size_t)rows*sizeof(float));

	cudaError err = cudaSetDevice(CUDA_DEVICE);
	if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	
	cudaMalloc((float**)&dev_a, vec_Size*sizeof(float));
	cudaMalloc((float**)&dev_b, size*sizeof(float));
	cudaMalloc((float**)&dev_partial_c, rows*sizeof(float));
	
    
    fp = fopen(argv[3], "r");
	if (fp == NULL) {
    		printf("Cannot Open the File");
		return 0;
	}

    
	
int i=0;
int j = 0;

	for(i = 0; i < rows; i++)
  		{
      		for(j = 0; j < cols; j++) 
      		{
				   fscanf(fp, "%f", &file_data);
				   mat[i][j] = file_data;
				//    printf("date=%f",mat[i][j]);
			  }
			//   printf("\n");

		  }
		  fclose(fp);
		 


	

	  
	  for(int i= 0; i < cols; i++)
	  {
		for(int j = 0; j < rows; j++)
		  {   
			  b[rows*i+j] = mat[j][i];
			  
			  
		  }
		//   printf("%f ", b[5]);
		}

		fv = fopen(argv[4], "r");
		  
		  
		  
					for(int j = 0; j < cols; j++) 
					{
						 fscanf(fv, "%f", &w_vect[j]);
						//  printf("v=%f",w_vect[j]);
					}
	  
	  fclose(fv);
		
	
	cudaMemcpy(dev_a, w_vect, vec_Size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size*sizeof(float), cudaMemcpyHostToDevice);
	
	int jobs = cols;
	int BLOCKS = (jobs + threads - 1)/threads;

	kernel<<<BLOCKS, threads>>>(dev_a, dev_b, dev_partial_c, rows, cols);

	cudaMemcpy(partial_c,dev_partial_c, rows*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<rows; i++) {
        printf("%f\n", partial_c[i]);
    }
    

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	free(partial_c);
}