/*
 ============================================================================
 Name        : nn.cu
 Author      : aDALOGLOU
 Version     : 1
 Copyright   : Adaloglou
 Description : CUDA compute reciprocals
 ============================================================================
 */
typedef struct {
int width;
int height;
float* elements;
} Matrix;
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#define ARRAYLENGTH(A) (sizeof(A)/sizeof(*(A)))
#define INPUT_VECTOR 10
#define NEURONS_L1 20		//MAX 1024
#define NEURONS_L2 5		//MAX 1024
#define NEURONS_L3 3		//max1024
#define LAYERS 3 // TODO use for more general form
#define MAX_NEURON_INIT_WEIGHT 0.8
///////////////////////////////////////////////////////////////////////////////////////////////
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
Matrix *matrixNew(int width, int height);
void matrixDestroy(Matrix *matrix);
void matrixInit ( Matrix *R  ) ;
void matrixPrint ( const Matrix *R  );
void vectorPrint( float *A , size_t length  ) ;
void ShowDevice(void);
float randf(float low,float high);
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float Sigmoid( float x )
{
	//if (x>=3.5) { return 0.95555555;  }//ask!!!!!!!!!!
	return ( (1)/(1+expf(-x))  ) ;
}
__device__ float Sigmoid_Derivative( float x )
{
	float s = (1)/(1+expf(-x)) ;
	return ( (s*(1-s))  ) ;
}

__device__ float HyperTan( float x )
{
	return ( (1-expf((-2)*x))/(1+expf(2*x))  ) ;
}
__device__ float HyperTan_Derivative( float x )
{
	float s=  ( (1-expf((-2)*x))/(1+expf(2*x))  ) ;
	return ((s+1)*(1-s));
}
/*kernel description
 * A is the vector of the output from the previous level
 * W is the weights of he current level
 * Out is the output vector to the next level of the NN
 *
 * */
__global__ void Kernel( float *A ,  Matrix W , float *Out ) // vector_size = W.width
{
	int tx =  threadIdx.x;
	float register sum = 0 ;
	int register k ;
	float a,b ;

//each thread multiplies one row of W with the input Vector A
#pragma unroll
	for(k=0 ; k<W.width  ; k++)
	{
		a = W.elements[ tx*W.width + k] ;
		b = A[k]   ;
		sum  += a*b ;
	}
	float output =  Sigmoid(sum) ;
	float derivative = Sigmoid_Derivative(sum) ;
	Out[tx]= output ;
}

float Feedforward(float *A , Matrix W1 , float *Out_1 ,  Matrix W2 , float *Out_2 , Matrix W3 , float *Out_3 )
{
	size_t  vector_in_size  = W1.width ;
	unsigned int neurons_L1 = W1.height ;
	unsigned int neurons_L2 = W2.height ;
	unsigned int neurons_L3 = W3.height ;

//measure time   TODO FUNCTION START STOP TIMER
	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	CUDA_CHECK_RETURN(cudaEventRecord(start));//starts the timer


	size_t sizeW1 = W1.width * W1.height * sizeof(float);
	Matrix d_W1;
	d_W1.width = W1.width;
	d_W1.height = W1.height;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_W1.elements, sizeW1));
	CUDA_CHECK_RETURN(cudaMemcpy(d_W1.elements, W1.elements, sizeW1,cudaMemcpyHostToDevice));

	float *d_A ;
	size_t vector_in_bytes =  (vector_in_size)* (sizeof(float)  ) ;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_A , vector_in_bytes ));
	CUDA_CHECK_RETURN(cudaMemcpy(d_A, A, vector_in_bytes ,cudaMemcpyHostToDevice));


	float *d_Out1;
	size_t vector_out1_bytes = (neurons_L1)*(sizeof(float)  ) ;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Out1 , vector_out1_bytes ));


	size_t sizeW2 = W2.width * W2.height * sizeof(float);
	Matrix d_W2;
	d_W2.width = W2.width;
	d_W2.height = W2.height;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_W2.elements, sizeW2));
	CUDA_CHECK_RETURN(cudaMemcpy(d_W2.elements, W2.elements, sizeW2,cudaMemcpyHostToDevice));

	float *d_Out2;
	size_t vector_out2_bytes = (neurons_L2)*(sizeof(float)  ) ;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Out2 , vector_out2_bytes ));
//////////////////////////////////////////////////////////////////////////////////////////////////
//level 3 allocation
	size_t sizeW3 = W3.width * W3.height * sizeof(float);
	Matrix d_W3;
	d_W3.width = W3.width;
	d_W3.height = W3.height;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_W3.elements, sizeW3));
	CUDA_CHECK_RETURN(cudaMemcpy(d_W3.elements, W3.elements, sizeW3,cudaMemcpyHostToDevice));

	float *d_Out3;
	size_t vector_out3_bytes = (neurons_L3)*(sizeof(float)  ) ;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Out3 , vector_out3_bytes ));

	//kernel_L1 call here

	Kernel<<< 1 , neurons_L1  >>>( d_A ,  d_W1 , d_Out1 ) ;
	cudaDeviceSynchronize();
	cudaMemcpy( Out_1 , d_Out1, vector_out1_bytes , cudaMemcpyDeviceToHost);


	Kernel<<< 1 , neurons_L2  >>>( d_Out1 ,  d_W2 ,  d_Out2  ) ;
	cudaDeviceSynchronize();
	cudaMemcpy( Out_2 , d_Out2, vector_out2_bytes , cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	vectorPrint( Out_2 , neurons_L2  ) ;   //debug

	//kernel level 3 call

	Kernel<<< 1 , neurons_L3  >>>( d_Out2 ,  d_W3 ,  d_Out3  ) ;
	cudaDeviceSynchronize();
	cudaMemcpy( Out_3 , d_Out3, vector_out3_bytes , cudaMemcpyDeviceToHost);
	vectorPrint( Out_3 , neurons_L3  ) ;   //debug

	//measure time
	cudaEventSynchronize(stop);// important!!!!!
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);


	//free device memory
	cudaFree( d_A ) ;
	cudaFree( d_W1.elements ) ;
	cudaFree( d_Out1 ) ;
	cudaFree( d_W2.elements ) ;
	cudaFree( d_Out2 ) ;
	cudaFree( d_W3.elements ) ;
	cudaFree( d_Out3 ) ;

	printf("Done. Time = %f\n",milliseconds  );
	return milliseconds ;
}

int main(void)
{
	ShowDevice();
	Matrix *W_L1 , *W_L2 , *W_L3  ;
	//float  A[INPUT_VECTOR] = { 1.0  , 0.2 , 0.8  } ;
	float  A[INPUT_VECTOR] = { 1.0  , 0.2 , 0.8  , 0.35 , 0.22 , 0.11 , 0.45 , 0.88 , 0.66 , 0.52 } ;
	//TODO general form maybe a function void  generateVector(INPUT_VECTOR)
	//TODO ADD BIAS 1 IN INPUT ARRAY
	size_t vector_in_size = ARRAYLENGTH(A);

	float Out_L1[NEURONS_L1] = { 0 } ;// static ?
	float Out_L2[NEURONS_L2] = { 0 } ;
	float Out_L3[NEURONS_L3] = { 0 } ;

	W_L1 = matrixNew( (int)vector_in_size , NEURONS_L1 ) ;// (width , height )
	matrixInit(W_L1) ;
	//matrixPrint( W_L1 );	// debug

	W_L2 = matrixNew( (int)NEURONS_L1 , NEURONS_L2  ) ;// (width , height )=(cols,rows)
	matrixInit(W_L2) ;
	//matrixPrint( W_L2 );	// debug


	W_L3 = matrixNew( (int)NEURONS_L2 , NEURONS_L3  ) ;// (width , height )=(cols,rows)
	matrixInit(W_L3) ;
	//matrixPrint( W_L3 );	// debug

	float gputime = Feedforward( A ,  *W_L1 ,  Out_L1 , *W_L2 , Out_L2 , *W_L3 , Out_L3  ) ;


	matrixDestroy(W_L1);
	matrixDestroy(W_L2);
	return 0;
}

void matrixDestroy(Matrix *matrix)
{
	free(matrix->elements);
	free(matrix);
}
void matrixInit ( Matrix *R  )
{
	//THE RANDOM NUMBER WIIL BE in range between [-MAX_NEURON_WEIGHT and  MAX_NEURON_WEIGHT]
	float MAXWEIGHT = MAX_NEURON_INIT_WEIGHT ;//Absolute value
	srand( time(0) );
	if (R)
	{
		for (int i=0 ; i<R->height; i++)
		{
			for (int j=0 ; j<R->width; j++)
				//R->elements[i*R->width +j]= (float)((2*MAX_NEURON_WEIGHT)*((double )rand() / (double)(RAND_MAX ) ))+(BIAS);//
				//R->elements[i*R->width +j]=1; // debug only
				R->elements[i*R->width +j]= (float)randf( (-1)*MAXWEIGHT , MAXWEIGHT ) ;
		}
	}
}
void matrixPrint ( const Matrix *R  )
{
#pragma unroll
	for (int i=0 ; i<R->height; i++)
			{
			for (int j=0 ; j<R->width; j++)
				printf(" i=%d ,j=%d ,   array.elements[i,j]=%f \n", i ,j, R->elements[i*R->width +j] ) ;
			}

}
Matrix *matrixNew(int width, int height)
{
	Matrix *M = (Matrix*)malloc(sizeof(Matrix));
	if (!M) return 0;

	M->height = height;
	M->width = width;

	int dimensions = ((M->height) * (M->width));
	size_t size = dimensions *  sizeof(float);

	M->elements = (float *) malloc( size );
	if(!M->elements) {
		free(M);
		return 0;
	}

	return M;
}
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
void vectorPrint( float *A , size_t length  )
{
	int i ;

	for( i=0 ; i <(int)(length) ;  i ++ )
	{

		std::cout<<"Arr[" << i << "]= " << A[i] <<std::endl;
	}


}
void ShowDevice(void)
{
	printf("NN Using CUDA - Starting...\n");

		int devID = 0;
	    cudaError_t error;
	    cudaDeviceProp deviceProp;

	    error = cudaGetDevice(&devID);

	    if (error != cudaSuccess)
	    {
	        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    }

	    error = cudaGetDeviceProperties(&deviceProp, devID);

	    if (deviceProp.computeMode == cudaComputeModeProhibited)
	    {
	        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
	        exit(EXIT_SUCCESS);
	    }

	    if (error != cudaSuccess)
	    {
	        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    }
	    else
	    {
	        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	    }



}
float randf(float low,float high){
    return (rand()/(double)(RAND_MAX))*abs(low-high)+low;
}
