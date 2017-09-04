#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "cuda.h"
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
int width;
int height;
float* elements;
} Matrix;

#define BLOCK_SIZE 32 //max 32
#define MULTIPLIER 32//max 128 (tested)
#define RANDMAX   10000
//function prototypes
Matrix *matrixNew(int width, int height);
void matrixDestroy(Matrix *matrix);
void matrixInit ( Matrix *R  ) ;
void matrixPrint ( const Matrix *R  );
float MatMulCpu(const Matrix A , const Matrix B , Matrix C  ) ;
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
float MSEError( const Matrix A , const Matrix B ) ;


float MatMul(const Matrix A, const Matrix B, Matrix C)
{	//measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);//starts the timer
///////////////////////////////////////////////////
	cudaError_t error;
	size_t size = A.width * A.height * sizeof(float);
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;

	cudaMalloc((void**)&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	error = cudaMalloc( (void**)&d_B.elements, size);
	error = cudaMemcpy(d_B.elements, B.elements, size,cudaMemcpyHostToDevice);

	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	error = cudaMalloc(  (void**) &d_C.elements  , size);

///////////////////////////////////////////////////////

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(MULTIPLIER,MULTIPLIER);
	//dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	for ( int w=0 ; w<10 ; w++   ){
		MatMulKernel<<< dimGrid , dimBlock >>>(d_A, d_B, d_C);
		cudaDeviceSynchronize();
	}

	cudaEventRecord(stop);

	cudaDeviceSynchronize();
	printf("Done\n");
	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	milliseconds=milliseconds/10 ;

	// Free device memory
	error = cudaFree( d_A.elements );
	error = cudaFree( d_B.elements );
	error = cudaFree( d_C.elements );
	int dim = BLOCK_SIZE*MULTIPLIER ;
	printf("Matrix Multiplication of two matrices dimension %d X %d in GPU time : %f milliseconds\n",dim,dim , milliseconds) ;
	return milliseconds ;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y; //row
	int j = blockIdx.x * blockDim.x + threadIdx.x; //col
	float register Cvalue = 0;
	int register  k;
	float a,b;
#pragma unroll
	for ( k=0 ; k<A.width   ; k++  )
	{
		a = A.elements[ i*A.width + k] ;
		b = B.elements[ k*B.width + j] ;
		Cvalue += a*b ;
	}
	C.elements[i * C.width + j] = Cvalue;

}


int main(void)
{
	srand(time(0));
	printf("Matrix Multiply Using CUDA - Starting...\n");
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

	Matrix *A, *B, *C , *D ;

	//allocate and initialization
	int dim = BLOCK_SIZE *MULTIPLIER ;

	A = matrixNew(dim,dim) ;
	B = matrixNew(dim,dim) ;
	C = matrixNew(dim,dim) ;
	D = matrixNew(dim,dim) ;

	matrixInit(A);
	matrixInit(B);

	float gputime = MatMul( *A , *B , *C ) ;
	float cputime = MatMulCpu( *A , *B , *D ) ;
	float speedup = (cputime/gputime) ;

	//matrixPrint(C) ;
	//matrixPrint(D) ;

	float mserror = MSEError( *C , *D ) ;
	printf("error = %f \t speedup = %f \n",mserror , speedup ) ;
	//////////////////////////
	// free matrices
	matrixDestroy(A);
	matrixDestroy(B);
	matrixDestroy(C);
	matrixDestroy(D);

	return 0;
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
void matrixDestroy(Matrix *matrix)
{
	free(matrix->elements);
	free(matrix);
}
void matrixInit ( Matrix *R  )
{

	if (R)
	{
		for (int i=0 ; i<R->height; i++)
		{
			for (int j=0 ; j<R->width; j++)
				R->elements[i*R->width +j]=(rand() % RANDMAX ) ;
		}
	}
}
void matrixPrint ( const Matrix *R  )
{
#pragma unroll
	for (int i=0 ; i<R->height; i++)
			{
			for (int j=0 ; j<R->width; j++)
				printf(" i=%d ,j=%d ,   array.elements[i,j]=%f \n", i ,j, R->elements[i*R->width +j] ) ;  ;
			}

}
float MatMulCpu(const Matrix A , const Matrix B , Matrix C  )
{
	clock_t start = clock();

	float register sum ;
	float register a,b ;
#pragma unroll
	for (int i=0 ; i<A.height; i++)
	{
			for (int j=0 ; j<A.width; j++)
			{
				sum = 0;
				for (int register k=0 ; k<A.width ; k++ )
				{
					a = A.elements[ i*A.width + k] ;
					b = B.elements[ k*B.width + j] ;
					sum += a*b ;
				}
				C.elements[i * C.width + j] = sum;
			}

	}

	clock_t end = clock();
	double dif = (double)(end - start) / CLOCKS_PER_SEC;
	 dif = dif*1000 ; // milisec
	printf ("CPU Your calculations took %.4lf milliseconds to run.\n", dif );
	return dif ;
}
float MSEError( const Matrix A , const Matrix B )
{

	float register error=0.0 ;
	float register a,b ;
	#pragma unroll
	for (int i=0 ; i<A.height; i++)
	{
		for (int j=0 ; j<A.width; j++)
		{
			a = A.elements[ i*A.width + j] ;
			b = B.elements[ i*B.width + j] ;
			error  += sqrt (abs(a-b)) ;

		}

	}
	return error ;

}
