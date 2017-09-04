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
#define INPUT_VECTOR 2048 //max 2048
#define NEURONS_L1 128	//MAX 1024



#define NEURONS_L2 200	//MAX 1024
#define NEURONS_L3 2		//MAX 1024
#define LAYERS 3			// TODO use for more general form
#define MAX_NEURON_INIT_WEIGHT 0.2
#define LRATE 1

///////////////////////////////////////////////////////////////////////////////////////////////
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
Matrix *matrixNew(int width, int height);
void matrixDestroy(Matrix *matrix);
void matrixInit ( Matrix *R  ) ;
void matrixPrint (  Matrix *R  );
void matrixPrint_HEADER (  Matrix *R , int x  ) ;
void vectorPrint( float *A , size_t length  ) ;
void ShowDevice(void);
float randf(float low,float high);
float SError_vec( float *A , float *B , size_t length) ;
int nextPowOf2( int x);
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float Sigmoid( float x )
{
	return ( (1)/(1+expf(-x))  ) ;
}

__device__ float Sig_der( float sigmoid_output )
{
	float s = sigmoid_output ;
	return ( (s*(1-s))  ) ;
}

__global__ void Kernel_FORWARD_OLD( float *A ,  Matrix W , float *Out ) // vector_size = W.width
{
	int tx = blockIdx.x ;
	float register sum = 0 ;
	int register k ;
	float register a,b ;
//each thread multiplies one row of W with the input Vector A
#pragma unroll
	for(k=0 ; k<W.width  ; k++)
	{
		a = W.elements[ tx*W.width + k] ;
		b = A[k]   ;
		sum  += ( a * b ) ;
	}
	float output =  Sigmoid(sum) ;
	Out[tx]= output ;
}


__global__ void Kernel_FORWARD_NEW( float *A ,  Matrix W , float *Out )
{
	extern __shared__ float sdata[];
	int bx = blockIdx.x ;
	unsigned int tid =  threadIdx.x;
	//unsigned int i = blockIdx.x*(W.width)+ threadIdx.x;
	//sdata[tid] =  (W.elements[i])*A[tid]
	unsigned int i = blockIdx.x*(W.width)+ threadIdx.x*2;

	double r1 = (W.elements[i])*A[tid] ;
	double r2 = (W.elements[i+1])*A[tid+1];
	sdata[tid] =   r1+ r2 ;
	// ATTEMPT TO MAKE FIRST ADD WHILE LOAD DATA - NOT WORKED YET
	//unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	//sdata[tid] =  (W.elements[i])*A[tid] +  (W.elements[i+blockDim.x])*A[tid] ;

	__syncthreads();
	/////////////////////////////
	// REDUCE HERE IN SHARED MEMORY- full unrool of loop
	// do reduction in shared mem
	if (W.width >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (W.width >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (W.width >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
	if (W.width >= 64) sdata[tid] += sdata[tid + 32];
	if (W.width >= 32) sdata[tid] += sdata[tid + 16];
	if (W.width >= 16) sdata[tid] += sdata[tid + 8];
	if (W.width >= 8) sdata[tid] += sdata[tid + 4];
	if (W.width >= 4) sdata[tid] += sdata[tid + 2];
	if (W.width >= 2) sdata[tid] += sdata[tid + 1];
	}
	// write result for this block to global mem
	if (tid == 0) Out[bx] = Sigmoid(sdata[0]);
}


__global__ void Kernel_forward_fast2( float *A ,  Matrix W , float *Out )
{
	extern __shared__ float sdata[];
	int bx = blockIdx.x ;
	unsigned int tid =  threadIdx.x;
	unsigned int i = blockIdx.x*(W.width)+ threadIdx.x;
	if ( tid<(W.width) )
	{
		sdata[tid] =  ((W.elements[i])*A[tid]);
	}
	else
	{
		sdata[tid] = 0 ; // padding in order to be power of 2 for reduction
	}
	__syncthreads();
	//******************* blockDim.x is power of 2 , Not W.width !!!!!!!
	// REDUCE HERE IN SHARED MEMORY- full unroll of for loop - reduction in shared mem
	if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32)
	{
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}
	// write result for this block to global mem
	if (tid == 0) Out[bx] = Sigmoid(sdata[0]);
}
// only for even number of inputs(W.width) and bigger than 128!
__global__ void Kernel_forward_fast3( float *A ,  Matrix W , float *Out )
{
	extern __shared__ float sdata[];
	int bx = blockIdx.x ;
	unsigned int tid =  threadIdx.x;
	unsigned int i = blockIdx.x*(W.width)+ threadIdx.x*2;
	if ( 2*tid<(W.width) )
	{
		sdata[tid] =  ((W.elements[i])*A[tid])+((W.elements[i+1])*A[tid+1]); // first reduction add during load
	}
	else
	{
		sdata[tid] = 0 ; // padding in order to be power of 2 for reduction
	}
	__syncthreads();
	//******************* blockDim.x is power of 2 , Not W.width !!!!!!!
	// REDUCE HERE IN SHARED MEMORY- full unroll of for loop - reduction in shared mem
	if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32)
	{
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}
	// write result for this block to global mem
	if (tid == 0) Out[bx] = Sigmoid(sdata[0]);
}


__global__ void Kernel_BACKPROP( float *A ,  Matrix W , float *Out_ff , float *Desired , float *Delta_val , float lrate )
{
    int tx =  threadIdx.x;
    float register DeltaW = 0 ;
    int register k ;

    float error = Out_ff[tx]-Desired[tx];
    float sigmoid_derivative= Sig_der( Out_ff[tx] ) ;
    float register delta_neuron = error*sigmoid_derivative ;

    Delta_val[tx] = delta_neuron ;

    float register w_old , w_new, previous_level_output ;
#pragma unroll
    for(k=0 ; k<W.width  ; k++)
    {
        w_old = W.elements[ tx*W.width + k] ;
        previous_level_output = ( ( k==(W.width-1) ) ? 1 :  A[k] )  ;
        DeltaW = delta_neuron * previous_level_output * lrate ;
        w_new = w_old - DeltaW ;
        W.elements[ tx*W.width + k] = w_new;
    }
}



__global__ void Kernel_BACKPROB_NEW( float *A ,  Matrix W , float *Out_ff , float *Desired , float *Delta_val , float lrate )
{
	unsigned int tid =  threadIdx.x;
    unsigned int bx = blockIdx.x ;
    float register DeltaW = 0 ;
    float register w_old , w_new, previous_level_output ;
    float error = Out_ff[bx]-Desired[bx];
    float sigmoid_derivative= Sig_der( Out_ff[bx] ) ;
    float register delta_neuron = error*sigmoid_derivative ;

    Delta_val[bx] = delta_neuron ;
    __syncthreads() ;
	w_old = W.elements[ bx*W.width + tid ] ;
	previous_level_output =  A[tid]   ;
	DeltaW = delta_neuron * previous_level_output * lrate ;
	w_new = w_old - DeltaW ;
	W.elements[ bx*W.width + tid] = w_new;
}

float Feedforward(float *A , Matrix *W1 , float *Out_1 ,  float *Out_2 )
{
//////////////////////////////////////////////////////////////////////////////////////////////////
// 1. memory transfers and  allocations on gpu memory
	size_t  vector_in_size  = W1->width ;
	unsigned int neurons_L1 = W1->height ;

	cudaEvent_t start, stop , start2 , stop2 , start3 , stop3;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	size_t sizeW1 = W1->width * W1->height * sizeof(float);
	Matrix d_W1;
	d_W1.width = W1->width;
	d_W1.height = W1->height;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_W1.elements, sizeW1));
	CUDA_CHECK_RETURN(cudaMemcpy(d_W1.elements, W1->elements, sizeW1,cudaMemcpyHostToDevice));

	float *d_A ;
	size_t vector_in_bytes =  (vector_in_size)* (sizeof(float)  ) ;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_A , vector_in_bytes ));
	CUDA_CHECK_RETURN(cudaMemcpy(d_A, A, vector_in_bytes ,cudaMemcpyHostToDevice));


	float *d_Out1;
	size_t vector_out1_bytes = (neurons_L1)*(sizeof(float)  ) ;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Out1 , vector_out1_bytes ));

	float *d_delta_val1;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_delta_val1 , vector_out1_bytes ));
	float Desired[neurons_L1] ;
	for ( int i=0 ; i< neurons_L1 ; i++  )  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		{
			Desired[i] = 1 ;       //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		}
	//float Delta1[neurons_L1] ;
	float *d_desired;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_desired , vector_out1_bytes ));
	CUDA_CHECK_RETURN(cudaMemcpy(d_desired, Desired , vector_out1_bytes ,cudaMemcpyHostToDevice));

//OLD KERNEL
	float milliseconds = 0;
	cudaEventRecord(start);
	for(int ww=0 ; ww<500 ; ww++)
	{
		Kernel_FORWARD_OLD<<< neurons_L1 , 1 >>>( d_A ,  d_W1 , d_Out1 ) ;
	}
	cudaEventRecord(stop);
	cudaMemcpy( Out_1 , d_Out1, vector_out1_bytes , cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

// NEW OPTIMIZED KERNEL
/*
	float milliseconds2 = 0;

	int threadsPerBlock = nextPowOf2( W1->width ) ;
	cudaEventRecord(start2);
	for(int ww=0 ; ww<500 ; ww++)
	{
		Kernel_forward_fast2<<< neurons_L1 , threadsPerBlock , (threadsPerBlock*sizeof(float)) >>>( d_A ,  d_W1 , d_Out1 ) ;
	}
	cudaEventRecord(stop2);
	cudaMemcpy( Out_2 , d_Out1, vector_out1_bytes , cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&milliseconds2, start2, stop2);
*/
//new 3
	///*
	float milliseconds3 = 0;
	int threadsPerBlock = (nextPowOf2( W1->width ))/2 ;
	cudaEventRecord(start3);
	for(int ww=0 ; ww<500 ; ww++)
	{
		Kernel_forward_fast3<<< neurons_L1 , threadsPerBlock , (threadsPerBlock*sizeof(float)) >>>( d_A ,  d_W1 , d_Out1 ) ;
	}
	cudaEventRecord(stop3);
	cudaMemcpy( Out_2 , d_Out1, vector_out1_bytes , cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&milliseconds3, start3, stop3);
//*/

	float error = SError_vec( Out_2 , Out_1 , neurons_L1 ) ;
	printf("\n\n error = %f \n\n\n", error );
	printf("time1 = %f milisec \n time2 = %f milisec \n speedup=%f\n" ,milliseconds, milliseconds3, (milliseconds/milliseconds3) );
	//printf("time1 = %f milisec \n time2 = %f milisec \n speedup=%f\n" ,milliseconds, milliseconds2, (milliseconds/milliseconds2) );

	printf("\n %f \t %f \t %f\n" ,milliseconds, milliseconds3 , (milliseconds/milliseconds3) );
	//printf("\n %f \t %f \t %f\n" ,milliseconds, milliseconds2 , (milliseconds/milliseconds2) );


	cudaFree( d_A ) ;
	cudaFree( d_W1.elements ) ;
	cudaFree( d_Out1 ) ;
	cudaFree( d_delta_val1 ) ;
	printf("\n Done."  );
	return milliseconds ;
}

int main(void)
{
	ShowDevice();
	Matrix *W_L1  ;
	float  A[INPUT_VECTOR]  ;
	for (int i =0 ; i <INPUT_VECTOR ; i++)	A[i]= 1 ;


	size_t vector_in_size = ARRAYLENGTH(A);
	float Out_L1[NEURONS_L1] = { 0 } ;
	float Out_L2[NEURONS_L1] = { 0 } ;

	W_L1 = matrixNew( (int)vector_in_size , NEURONS_L1 ) ;// (width , height )==(PREVIOUS NEURONS , CURRENT LAYER NEURONS)
	matrixInit(W_L1) ;

	float gputime = Feedforward( A ,  W_L1 ,  Out_L1 ,  Out_L2  ) ;
	matrixDestroy(W_L1);
	return 0;
}

void matrixDestroy(Matrix *matrix)
{
	free(matrix->elements);
	free(matrix);
}


float SError_vec( float *A , float *B , size_t length)
{

	float register error=0.0 ;
	float register a,b ;

	#pragma unroll
	for (int i=0 ; i<length ;  i++)
	{
			a = A[i] ;
			b = B[i] ;
			error  += 0.5*(a-b)*(a-b) ;
	}

	return error ;
}

void matrixInit ( Matrix *R  )
{
	srand( time(0) );
	if (R)
	{
		for (int i=0 ; i<R->height; i++)
		{
			for (int j=0 ; j<R->width; j++)
				//R->elements[i*R->width +j]= (float)((2*MAX_NEURON_WEIGHT)*((double )rand() / (double)(RAND_MAX ) ))+(BIAS);//
				R->elements[i*R->width +j]= 0.01 ; // debug only
				//R->elements[i*R->width +j]= (float)randf(  -0.1 , 0.1 ) ;
		}
	}
}
void matrixPrint (  Matrix *R  )
{
#pragma unroll
	for (int i=0 ; i<R->height; i++)
			{
			for (int j=0 ; j<R->width; j++)
				printf(" i=%d ,j=%d ,   array.elements[i,j]=%f \n", i ,j, R->elements[i*R->width +j] ) ;
			}

}
void matrixPrint_HEADER (  Matrix *R  , int x )
{

	int height = ( (R->height >= x)? x : R->height ) ;
	int width  =( (R->width >= x)? x : R->width ) ;

#pragma unroll
	for (int i=0 ; i<height; i++)
			{
			for (int j=0 ; j<width; j++)
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
	float len = (length<15)?length :15;
	for( i=0 ; i <(int)(len) ;  i ++ )
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

int nextPowOf2( int x)
{
	if ( !(x & (x - 1)) ) return x ; // is power of 2
	int power = 4 ;
	while ( power<x )
	{
		power*=2 ; // next power of two
	}
	return power ;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
	test for possible otimazation to backpropagation kernel for output layer
	float mili3 , mili4 ;
		cudaEvent_t start3, stop3 , start4, stop4;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventCreate(&start4);
	cudaEventCreate(&stop4);

	float Delta1TEST[neurons_L1];
	cudaEventRecord(start3);
	for(int ww=0 ; ww<10000 ; ww++)
	Kernel66<<< neurons_L1 , W1->width  >>>( d_A ,  d_W1 ,  d_Out1 , d_desired , d_delta_val1 , 0.5 ) ;

	cudaEventRecord(stop3);
	CUDA_CHECK_RETURN(cudaMemcpy( W1->elements, d_W1.elements,sizeW1,cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy( Delta1, d_delta_val1 ,vector_out1_bytes ,cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&mili3, start3, stop3);


	cudaEventRecord(start4);

	for(int ww=0 ; ww<10000 ; ww++)
	Kernel2<<< 1 , neurons_L1  >>>( d_A ,  d_W1 ,  d_Out1 , d_desired , d_delta_val1 , 0.5 ) ;

	cudaEventRecord(stop4);
	CUDA_CHECK_RETURN(cudaMemcpy( W1->elements, d_W1.elements,sizeW1,cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy( Delta1TEST, d_delta_val1 ,vector_out1_bytes ,cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop4);
	cudaEventElapsedTime(&mili4, start3, stop3);

	float error2 = SError_vec( Delta1TEST , Delta1 , neurons_L1 ) ;
	printf("error = %f \n", error );
	printf("time1 = %f milisec \n time2 = %f milisec \n speedup=%f\n" ,mili3, mili4 , (mili4/mili3) );


	printf("-----\n Delta level 1 non optimized-------\n");
	vectorPrint( Delta1TEST  , neurons_L1 ) ;
	printf("-----\n Delta level 1--------\n");
	vectorPrint( Delta1  , neurons_L1 ) ;
	*/


