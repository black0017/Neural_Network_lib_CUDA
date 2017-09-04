/*
 ============================================================================
 Name        : scan.cu
 Author      : adaloglou
 Version     :
 Copyright   : ___Adaloglou___
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include<stdio.h>
#include<math.h>

#define SIZE 1024
typedef struct
{
	int width;
	int height;
	float* elements;
} Matrix;
void matrixInitSet ( Matrix *matrix  , float value  )
{

	if (matrix)
	{
		for (int i=0 ; i<matrix->height; i++)
		{
			for (int j=0 ; j<matrix->width; j++)
			{
				matrix->elements[i*matrix->width +j]=value ;
			}
		}
	}
}
void vectorPrint( float *A , size_t length  )
{
	size_t i ;
	for( i=0 ; i < length ;  i ++ )
	{
		printf("i= %lu value  = %f \n", i , A[i]);

	}
	printf("-----------------\n");
}


__global__ void reduce0(float *g_idata, float *g_odata)
{
	extern __shared__ float sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2)
	{
		if (tid % (2*s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}


__global__ void reduce1(float *g_idata, float *g_odata)
{
	extern __shared__ float sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=1; s < blockDim.x; s *= 2)
	{
		int index = 2 * s * tid;
		if (index < blockDim.x) {
		sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

__global__ void reduce3(float *g_idata, float *g_odata)
{
	extern __shared__ float sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
	if (tid < s)
	{
	sdata[tid] += sdata[tid + s];
	}
	__syncthreads();

	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce4(float *g_idata, float *g_odata)
{
	extern __shared__ float sdata3[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata3[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
	if (tid < s)
	{
		sdata3[tid] += sdata3[tid + s];
	}
	__syncthreads();

	}
	// write result for this block to global mem
	if (tid == 0)
		{

		g_odata[blockIdx.x] = sdata3[0];
		sdata3[0] =0 ;
		}

}


__global__ void reduce5(float *g_idata, float *g_odata)
{
	extern __shared__ float sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem

	for (unsigned int s=blockDim.x/2; s>32; s>>=1)
	{
	if (tid < s)
	sdata[tid] += sdata[tid + s];
	__syncthreads();
	}
	if (tid < 32)
	{
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
		{
		g_odata[blockIdx.x] = sdata[0];
		sdata[0]=0;
		}

}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n)
{
extern __shared__ float sdata2[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockSize*2) + tid;
unsigned int gridSize = blockSize*2*gridDim.x;
sdata2[tid] = 0;
while (i < n) { sdata2[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
__syncthreads();
if (blockSize >= 512) { if (tid < 256) { sdata2[tid] += sdata2[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata2[tid] += sdata2[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata2[tid] += sdata2[tid + 64]; } __syncthreads(); }
if (tid < 32) {
if (blockSize >= 64) sdata2[tid] += sdata2[tid + 32];
if (blockSize >= 32) sdata2[tid] += sdata2[tid + 16];
if (blockSize >= 16) sdata2[tid] += sdata2[tid + 8];
if (blockSize >= 8) sdata2[tid] += sdata2[tid + 4];
if (blockSize >= 4) sdata2[tid] += sdata2[tid + 2];
if (blockSize >= 2) sdata2[tid] += sdata2[tid + 1];
}
if (tid == 0) g_odata[blockIdx.x] = sdata2[0];
}


int main()
 {
    float h_in[SIZE],h_out[SIZE];
    int i;
     for (i = 0; i < SIZE; i++)
        h_in[i] = 1.0;
    //for (i = 0; i < SIZE; i++)	printf("%d ", h_in[i]);



   float *d_in2;
   float *d_out2;
   cudaMalloc((void**)&d_in2, sizeof(float)* SIZE);
   cudaMalloc((void**)&d_out2, sizeof(float)*SIZE);
   cudaMemcpy(d_in2, h_in, sizeof(float) * SIZE, cudaMemcpyHostToDevice);


/*
   reduce0<<<1, SIZE, (sizeof(float)*SIZE) >>>(d_in2,d_out2 );
   cudaMemcpy((void *)h_out, (void *)d_out2, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
   printf("\n R0 \n");
   printf(" value  = %f \n", h_out[0]);


   reduce1<<<1, SIZE, (sizeof(float)*SIZE) >>>(d_in2,d_out2 );
   cudaMemcpy((void *)h_out, (void *)d_out2, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
   printf("\n R1 \n");
   printf(" value  = %f \n", h_out[0]);


   reduce3<<<1, SIZE, (sizeof(float)*SIZE) >>>(d_in2,d_out2 );
   cudaMemcpy((void *)h_out, (void *)d_out2, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
   printf("\n R3 \n");
   printf(" value  = %f \n", h_out[0]);

  /////////////////////////////////////////////////////////////////
   reduce4<<<1, SIZE, (sizeof(float)*SIZE) >>>(d_in2,d_out2 );
   cudaMemcpy((void *)h_out, (void *)d_out2, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
   printf("\n  R4  \n ");
   vectorPrint( h_out , 1  ) ;
*/
     reduce5<<<1, SIZE, (sizeof(float)*SIZE) >>>(d_in2,d_out2 );
     cudaMemcpy((void *)h_out, (void *)d_out2, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
     printf("\n r5 \n");
     printf(" value  = %f \n", h_out[0]);


  return 0;
}



/*
 *
 * __global__ void scan(int *d_in,int *d_out,int n)
{
    extern __shared__ int sdata[];
    int i;
    int tid = threadIdx.x;
    sdata[tid] = d_in[tid];
    for (i = 1; i <n; i <<= 1)
    //for (i = 1; i <n; i++)
     {

        if (tid>=i)
         {
            sdata[tid] +=sdata[tid-i];
        }
        __syncthreads();
    }
    d_out[tid] = sdata[tid];
     __syncthreads();
 }

int main()
 {

    int h_in[SIZE],h_out[SIZE];
    int i,j;
     for (i = 0; i < SIZE; i++)
        h_in[i] = 2*i+1;
    for (i = 0; i < SIZE; i++)
        printf("%d ", h_in[i]);
   int *d_in;
   int *d_out;
   cudaMalloc((void**)&d_in, sizeof(int)* 16);
   cudaMalloc((void**)&d_out, sizeof(int)* 16);
   cudaMemcpy(d_in, h_in, sizeof(int) * 16, cudaMemcpyHostToDevice);

   scan <<<1, SIZE, sizeof(int)*SIZE >>>(d_in,d_out, SIZE);

   cudaMemcpy(h_out, d_out, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
   printf("\n\n");
   for (i = 0; i < SIZE; i++)
      printf("%d ", h_out[i]);
  return 0;
}


*/


