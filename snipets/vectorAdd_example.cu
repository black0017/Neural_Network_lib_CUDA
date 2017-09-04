
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define N 10000000

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}


int main(void){


	srand(time(0));
	cudaError_t err = cudaSuccess;
	 cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

			  cudaEventRecord(start);
	int numElements = N;

	size_t size = numElements * sizeof(float);

	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	/*/
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	/*/

	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}


	float *d_A = NULL;

	float *d_B = NULL;

	float *d_C = NULL;

	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	/*/////
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	///*/

	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements+threadsPerBlock-1 ) / threadsPerBlock;


	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);

	/*//////
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result not correct at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	//printf("Test PASSED\n");

	//*//////

	/*/

	//
	*/

	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Vector Addition in GPU time : %f milliseconds\n", milliseconds) ;

	err = cudaFree(d_A);

	err = cudaFree(d_B);

	err = cudaFree(d_C);


	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);



	printf("Done\n");
	return 0;
}

