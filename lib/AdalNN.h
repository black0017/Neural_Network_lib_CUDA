/*
 * AdalNN.h
 *
 *  Created on: Jan 3, 2017
 *      Author: nikolas
 */

#ifndef ADALNN_H_
#define ADALNN_H_

#include <cstdarg>
#include <cassert>
#include "AdalMatrix.h"
#define d0 0.1
enum  BPA{ CLASSIC=0 , RESILIENT , ADAPTIVE , MOMENTUM , QUICK  } ;

typedef struct NNState
{
	/* weights is an 'array' of pointers with the size of the levels of the NN
	 * for each level there is a pointer that point to a Matrix struct
	 * Matrix is a struct created to represent the weights of each level instead of
	 * 2D array : Array[rows=height][colums=width] ==  Matrix *matrixNew(int width, int height)
	 * Matrix struct also saves the dimensions  Matrix.height = height; M->width = width;
	 * this is needed in the gpu kernel.
	 * Height is the number of neurons of the current level
	 * Width is the output of the previous level , the neurons of the previous level
	 * or the input array for the first level
	 */
	Matrix **weights;
	Matrix **dweight;
	Matrix **grad , **dij_prev , **Lrate ;
	/* ffout is the feed forward output of the current level
	 * ffout is an array of pointers with the size of the levels
	 * each one points to the first neuron output of a specific level , so
	 * we have as floats as the size of the neurons of each level that is given from the user
	 *
	 *
	 */
	float **ffout;

	/* delta_val is the calculated delta value from the equation
	 * :  --------------------------
	 * it is used in the backpropagation and is a metric of the error for each neuron in the NN ,
	 * error is  the difference from the desired output of the NN
	 * delta_val is an dynamic array of pointers with the size of the levels
	 * each one points to the first delta value of a specific level and is the size of the neurons
	 * with delta values the output error propagates in the hidden levels of the NN and the
	 * weights are adjusted
	 */
	float **delta_val;

	/*
	 * neuron is a pointer that points to an integer which is the number of neurons for each layer
	 */
	int *neurons;

	size_t levels;

	size_t max_iterations ;
	/*
	* the Learning rate that is used for the backprobagation
	*/
	float  lrate ;
	int sampling ;
	//gpu pointers
	float *d_input_vec , *d_desired, **d_Out_ff ,**d_delta_val ;

	//float **d_data ;

	const static int maxData = 200 ;
	//gpu W matrix pointer
	Matrix *d_W ;

	/* back propagation variants data
	 *  d_Grad , dij_prev for resilient BPA
	 *  dW_prev    for momentum BPA
	 *  dLrate is used for adaptive learning BPA
	 */
	Matrix *d_Grad , *d_dij_prev , *dW_prev , *d_Lrate;
	BPA variant ;
} NNState;

NNState *initNNState( int input_vec_len, float Lrate ,size_t max_iter , int sampling  , BPA variant ,  size_t levels, ... );
void freeNNState(NNState *state);

void copyDataToGpu( NNState *state ) ;
void copyDataToCpu( NNState *state   );
unsigned int isPowOf2(unsigned int x);
unsigned int nextPowOf2( unsigned int x);
unsigned int isEven(unsigned int x);

float trainNNetwork_test(NNState *state, float *input, float *desired , int iteration  ) ;
float trainNNetwork_final(NNState *state, float *input, float *desired , int iteration  );
float evalNN( NNState *state, float *input, size_t input_len ,  int desired ) ;
void dbgPrintNNState(NNState *state);
void showInfo( NNState *state , int instances  ) ;
int NNvar(NNState *state) ;
void printBPA(BPA e);
#endif /* ADALNN_H_ */
