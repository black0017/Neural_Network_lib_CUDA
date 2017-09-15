/*
 * mat_func.h
 *
 *  Created on: Dec 30, 2016
 *      Author: nikolas
 */
#ifndef MAT_FUNC_H_
#define MAT_FUNC_H_
#include <time.h>
#include <cstdio>
#include <cstdlib>
//#include <omp.h>
#include <math.h>

#define randf(low, high) ((rand()/(double)(RAND_MAX))*abs(low-high)+low)

typedef struct
{
	unsigned int width;
	unsigned int height;
	float* elements;
} Matrix;

#define MAX_NEURON_INIT_WEIGHT 0.25


static void matrixPrint ( const Matrix *matrix  )
{
#pragma unroll
	for (int i=0 ; i< matrix->height; i++)
	{
		for (int j=0 ; j < matrix->width; j++)
			printf("array.elements[%d,%d] = %f \n", i ,j, matrix->elements[i*matrix->width +j] ) ;
	}
}


static void matrixPrint_HEADER (  Matrix *R  , int x )
{

	int height = ( (R->height >= x)? x : R->height ) ;
	int width  =( (R->width >= x)? x : R->width ) ;

#pragma unroll
	for (int i=0 ; i<height; i++)
			{
			for (int j=0 ; j<width; j++)
				printf("Weights [ %d , %d  ] = %f \n", i ,j, R->elements[i*R->width +j] ) ;
			}

}

static void matrixDestroy(Matrix *matrix)
{
	if (matrix)
	{
		if (matrix->elements)
			free(matrix->elements);
		free(matrix);
	}
}

static void matrixInit ( Matrix *matrix  )
{
	//THE RANDOM NUMBER WIIL BE in range between [-MAX_NEURON_WEIGHT and  MAX_NEURON_WEIGHT]

	float MAXWEIGHT = MAX_NEURON_INIT_WEIGHT  ;
	float MINWEIGHT =  (-1)*MAX_NEURON_INIT_WEIGHT   ;

	if (matrix)
	{
		for (int i=0 ; i<matrix->height; i++)
		{
			for (int j=0 ; j<matrix->width; j++)
			{
				//matrix->elements[i*matrix->width +j]=0.5; // debug only !!!
				matrix->elements[i*matrix->width +j]= randf( MINWEIGHT , MAXWEIGHT ) ;
			}
		}
	}
}


static void matrixInitSet ( Matrix *matrix  , float value  )
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


static Matrix *matrixNew(int width, int height)
{
	if (width <= 0 || height <= 0) return 0;

	Matrix *M = (Matrix*)malloc(sizeof(Matrix));
	if (!M) return 0;

	M->height = height;
	M->width = width;

	int dimensions = ((M->height) * (M->width));
	size_t size = dimensions *  sizeof(float);

	M->elements = (float *) malloc( size );
	if(!M->elements)
	{
		free(M);
		return 0;
	}

	return M;
}

static void vectorPrint( float *A , size_t length  )
{
	size_t i ;

	for( i=0 ; i < length ;  i ++ )
	{
		printf("i= %lu value  = %f \n", i , A[i]);
	}
	printf("-----------------\n");
}
static int vectorMax( float *A , size_t length  )
{
	size_t i ;
	float max=-200 ;
	int pos;
	for( i=0 ; i < length ;  i++ )
	{
		if (max < A[i])
		{
			max = A[i] ;
			pos = i ;
		}
	}
	return (pos);
}

static void vectorPrintINT( int *A , size_t length  )
{
	size_t i ;

	for( i=0 ; i < length ;  i ++ )
	{
		printf("i= %lu value  = %d \n", i , A[i]);
	}
	printf("-----------------\n");
}

static void vectorPrintINT_HEADER( int *A , size_t length  )
{
	size_t i ;

	for( i=0 ; i < 100 ;  i ++ )
	{
		printf("i= %lu value  = %d \n", i , A[i]);
	}
	printf("-----------------\n");
}

static float MSEError_mat( const Matrix A , const Matrix B )
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

static float SError_vec( float *A , float *B , size_t length)
{

	float register error=0.0 ;
	float register a,b ;

	#pragma unroll
	for (int i=0 ; i<length ;  i++)
	{
			a = A[i] ;
			b = B[i] ;
			error  += (a-b)*(a-b) ;
	}
	return (error/2) ;
}
static void print_array(const float *A, size_t width, size_t height)
{
  size_t i , j ;
  for( i = 0; i < height; i++)
  {
	  //printf("row %d results \n", (int)i);
    for( j = 0; j < width; j++)
    	printf("%.2f\t", A[i * width + j]);

    putchar('\n');
  }
}
static void print_arrayHeader(const float *A, size_t width, size_t height)
{
  size_t i , j ;
  for( i = 0; i < 30; i++)
  {
    for( j = 0; j < width; j++)
    	printf("%.2f\t", A[i * width + j]);

    putchar('\n');
    putchar('\n');
  }
  printf("----------------------------\n");
}
#endif /* MAT_FUNC_H_ */
