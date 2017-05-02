
#include <cstdio>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include "cuda.h"
#include "AdalNN.h"
#include "Kernels.h"
#include "limits.h"
#define OUTPUT_LAYER  ((state->levels)-1)
#define randf(low, high) ((rand()/(double)(RAND_MAX))*abs(low-high)+low)
#define d0 0.1
static const char *BPA_str[] = {   "CLASSIC",   "RESILIENT",   "ADAPTIVE",   "MOMENTUM",   "QUICK"};
void printBPA(BPA e){   printf("BPA = %s\n", BPA_str[(std::size_t)e]);}
//this value ensures that 'srand' is called only once
static bool RandIsInitialized = false;
//add bias neuron
static bool Bias = true;


//CUDA error check macro
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

NNState *initNNState( int input_vec_len, float Lrate , size_t max_iterations ,int  sampling , BPA variant , size_t levels, ... )
{
	if (!RandIsInitialized)
		{
			std::srand( std::time(0) );
			RandIsInitialized = true;
		}
	//if input vector length is negative bye bye
	if (input_vec_len < 0)
		return 0;
	if (levels <= 1  )
		return 0 ;

	//create new NNState object
	NNState *ret = (NNState *)malloc(sizeof(NNState));
	if (!ret)
		return 0;

	//allocate 'levels' neurons
	ret->neurons = (int *)malloc(sizeof(int) * levels);
	ret->max_iterations  = max_iterations ;
	ret->sampling = sampling ;
	ret->variant = variant ;
	if (!ret->neurons)
	{
		free(ret);
		ret = 0;
	}

	//get the number of neurons, of each level, from the arguments
	va_list neuron_list;
	va_start(neuron_list, levels);

	for(int i = 0; i < levels; i++) {
		ret->neurons[i] = va_arg(neuron_list, int);
	}

	va_end(neuron_list);
	ret->ffout = (float**)calloc(levels, sizeof(float*));
	assert(ret->ffout);
	ret->delta_val = (float**)calloc(levels, sizeof(float*));
	assert(ret->delta_val);
	ret->levels = levels;
	ret->lrate = Lrate ;
	//allocate dynamic array with size 'levels' of Matrix type
	ret->weights = (Matrix**)malloc(sizeof(Matrix*) * levels);
	assert(ret->weights);
	//device memory
	ret->d_input_vec = (float *)malloc(sizeof(float*) * ret->levels);
	ret->d_desired = (float *)malloc(sizeof(float*) * ret->levels);
	ret->d_Out_ff = (float **)malloc(sizeof(float*) * ret->levels);
	ret->d_delta_val = (float **)malloc(sizeof(float*) * ret->levels);
	ret->d_W = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);



	switch ( ret->variant )
	{
		case CLASSIC :
		{
			break ;
		}
		case MOMENTUM :
		{
			ret->dW_prev = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);//gpu
			ret->dweight = (Matrix**)malloc(sizeof(Matrix*) * ret->levels);//cpu
			break;
		}
		case RESILIENT :
		{
			//cpu
			ret->grad = (Matrix**)malloc(sizeof(Matrix*) * ret->levels);
			ret->dij_prev = (Matrix**)malloc(sizeof(Matrix*) * ret->levels);
			//gpu
			ret->d_Grad = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);
			ret->d_dij_prev = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);

			ret->dW_prev = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);//gpu
			ret->dweight = (Matrix**)malloc(sizeof(Matrix*) * ret->levels);//cpu

			break;
		}
		case ADAPTIVE :
		{
			ret->grad = (Matrix**)malloc(sizeof(Matrix*) * levels);
			ret->d_Grad = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);

			ret->Lrate = (Matrix**)malloc(sizeof(Matrix*) * levels);//cpu
			ret->d_Lrate = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);
			break;
		}
		case QUICK :
		{
			ret->dW_prev = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);//gpu
			ret->dweight = (Matrix**)malloc(sizeof(Matrix*) * levels);//cpu

			ret->grad = (Matrix**)malloc(sizeof(Matrix*) * levels);
			ret->d_Grad = (Matrix *)malloc(sizeof(Matrix *) * ret->levels);
		}


	}

	//optimization
	//ret->d_data = (float **)malloc(sizeof(float*) * ret->maxData);


	// allocate space for 1 training data
	size_t vector_in_bytes =  (input_vec_len)* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_input_vec , vector_in_bytes ));
	size_t target_bytes =  (ret->neurons[levels-1])* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_desired , target_bytes ));

	size_t sizeW , neurons , vector_out_bytes ;
	for (size_t i = 0; i < (levels); i++)
	{

		ret->ffout[i] = (float*)calloc(ret->neurons[i], sizeof(float));
		ret->delta_val[i] = (float*)calloc(ret->neurons[i] , sizeof(float));
		short int bias = (Bias==true)? 1 : 0 ;											/* WIDTH                              x HEIGHT          */
		ret->weights[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
		matrixInit( ret->weights[i] );
		sizeW = (ret->weights[i]->width )*( ret->weights[i]->height)  * sizeof(float);
		ret->d_W[i].width = ret->weights[i]->width;
		ret->d_W[i].height = ret->weights[i]->height;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_W[i].elements, sizeW));
		neurons = ret->neurons[i] ;
		vector_out_bytes = (neurons)*(sizeof(float)) ;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_delta_val[i] , vector_out_bytes ));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_Out_ff[i] , vector_out_bytes ));
		//----------------------------------------------------------------
		//-----------------------------------------------------------------
		switch ( variant )
			{
				case CLASSIC : break ;
				case MOMENTUM :
				{
					ret->dweight[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->dweight[i] , 0 );
					ret->dW_prev[i].width = ret->weights[i]->width;
					ret->dW_prev[i].height = ret->weights[i]->height;
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->dW_prev[i].elements, sizeW));
					break;
				}
				case RESILIENT :
				{
					//DW_PREV
					ret->dweight[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->dweight[i] , 0.001 );
					ret->dW_prev[i].width = ret->weights[i]->width;
					ret->dW_prev[i].height = ret->weights[i]->height;
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->dW_prev[i].elements, sizeW));
					//cpu
					ret->grad[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->grad[i] , 0.001 );
					ret->dij_prev[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->dij_prev[i] , d0 );
					//gpu
					ret->d_dij_prev[i].width = ret->weights[i]->width;
					ret->d_dij_prev[i].height = ret->weights[i]->height;
					ret->d_Grad[i].width = ret->weights[i]->width;
					ret->d_Grad[i].height = ret->weights[i]->height;
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_Grad[i].elements, sizeW));
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_dij_prev[i].elements, sizeW));
					break;
				}
				case ADAPTIVE :
				{
					ret->grad[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->grad[i] , 0 );
					ret->d_Grad[i].width = ret->weights[i]->width;
					ret->d_Grad[i].height = ret->weights[i]->height;
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_Grad[i].elements, sizeW));
					//cpu
					ret->Lrate[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->Lrate[i] , ret->lrate );
					//gpu
					ret->d_Lrate[i].width = ret->weights[i]->width;
					ret->d_Lrate[i].height = ret->weights[i]->height;
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_Lrate[i].elements, sizeW));
					break;
				}
				case QUICK :
				{
					ret->grad[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->grad[i] , 0 );
					ret->d_Grad[i].width = ret->weights[i]->width;
					ret->d_Grad[i].height = ret->weights[i]->height;
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_Grad[i].elements, sizeW));

					ret->dweight[i] = matrixNew( (i == 0) ? (input_vec_len+bias) : (ret->neurons[i-1]+bias) ,  ret->neurons[i] ) ;
					matrixInitSet( ret->dweight[i] , 0 );
					ret->dW_prev[i].width = ret->weights[i]->width;
					ret->dW_prev[i].height = ret->weights[i]->height;
					CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->dW_prev[i].elements, sizeW));
				}


			}

//---------------------------------------

	}
/*
	//allocate size for many data
	for (size_t i = 0; i < ret->maxData ; i++)
		{
		CUDA_CHECK_RETURN(cudaMalloc((void**)&ret->d_data[i] , vector_in_bytes ));
		}
*/
	return ret;
}

float trainNNetwork(NNState *state, float *input, float *desired , int iteration  )
{
	//TODO optimazation tranfer all the training data together in gpu to reach max bandwidth and then
	// then i can make idealy a device to device memcpy!!!!

	size_t vector_in_bytes =  ((state->weights[0]->width)-1)* (sizeof(float)); // width -1 for bias
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_input_vec, input , vector_in_bytes ,cudaMemcpyHostToDevice));
	size_t target_bytes =  (state->neurons[OUTPUT_LAYER])* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_desired, desired, target_bytes ,cudaMemcpyHostToDevice));

	//2.Feedforward kernel calls  that executed serially
	size_t i, neurons ;
	float total_error=-1 ;
	cudaDeviceSynchronize();
	for ( i = 0 ; i < (state->levels); i++)
	{
		neurons = state->neurons[i] ;

		if ( isPowOf2(state->d_W[i].width)  )
		{
			//optimized kernel - reduction
			Kernel_forward_fast<<< neurons ,state->d_W[i].width , (state->d_W[i].width*sizeof(float)) >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i] , state->d_Out_ff[i]  ) ;
		}
		else
		{
			Kernel_forward<<< 1 , neurons  >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ), state->d_W[i] ,  state->d_Out_ff[i] ) ;
		}

		cudaDeviceSynchronize();
	}
//copy back results to cpu to calculate error when needed
	if ( iteration%(state->sampling) == 0)
	{
		cudaMemcpy( state->ffout[OUTPUT_LAYER] , state->d_Out_ff[OUTPUT_LAYER], state->neurons[OUTPUT_LAYER]*(sizeof(float)) , cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		total_error  = SError_vec(  state->ffout[ OUTPUT_LAYER  ] , desired , state->neurons[ OUTPUT_LAYER ] );
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3. Back propagation Kernels call here
	float Lrate = state->lrate ;
	for ( i = OUTPUT_LAYER ;    i != SIZE_MAX    ; i--) //
	{
		neurons = state->neurons[i]   ;
		if (i == OUTPUT_LAYER )
		{
			Kernel_back_last<<< 1 , neurons >>>( state->d_Out_ff[i-1] ,  state->d_W[i] , state->d_Out_ff[i] , state->d_desired , state->d_delta_val[i]   , Lrate  ) ;
		}
		else
		{
			Kernel_back_hidden<<< 1, neurons >>>(  ( (i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i]  , state->d_Out_ff[i] , state->d_W[i+1]  , state->d_Out_ff[i+1] , state->d_delta_val[i]  , 	 state->d_delta_val[i+1] ,  Lrate	) ;
		}
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	return total_error ;

}


float trainNNetwork_momentum(NNState *state, float *input,  float *desired , int iteration  )
{
	size_t vector_in_bytes =  ((state->weights[0]->width)-1)* (sizeof(float)); // width -1 for bias
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_input_vec, input , vector_in_bytes ,cudaMemcpyHostToDevice));
	size_t target_bytes =  (state->neurons[OUTPUT_LAYER])* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_desired, desired, target_bytes ,cudaMemcpyHostToDevice));

	//2.Feedforward kernel calls  that executed serially
	size_t i, neurons ;
	float total_error=-1 ;
	cudaDeviceSynchronize();
	for ( i = 0 ; i < (state->levels); i++)
	{
		neurons = state->neurons[i] ;

		if ( isPowOf2(state->d_W[i].width)  )
		{
			//optimized kernel - reduction
			Kernel_forward_fast<<< neurons ,state->d_W[i].width , (state->d_W[i].width*sizeof(float)) >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i] , state->d_Out_ff[i]  ) ;
		}
		else
		{
			Kernel_forward<<< 1 , neurons  >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ), state->d_W[i] ,  state->d_Out_ff[i] ) ;
		}

		cudaDeviceSynchronize();
	}
//copy back results to cpu to calculate error when needed
	if ( iteration%(state->sampling) == 0)
	{
		cudaMemcpy( state->ffout[OUTPUT_LAYER] , state->d_Out_ff[OUTPUT_LAYER], state->neurons[OUTPUT_LAYER]*(sizeof(float)) , cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		total_error  = SError_vec(  state->ffout[ OUTPUT_LAYER  ] , desired , state->neurons[ OUTPUT_LAYER ] );
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3. Back propagation Kernels call here
	float Lrate = state->lrate ;
	for ( i = OUTPUT_LAYER ;    i != SIZE_MAX    ; i--) //
	{
		neurons = state->neurons[i]   ;
		if (i == OUTPUT_LAYER )
		{
			Kernel_momentum_last<<< 1 , neurons >>>( state->d_Out_ff[i-1] ,  state->d_W[i] , state->d_Out_ff[i] , state->d_desired , state->d_delta_val[i]   , Lrate ,iteration ,  state->dW_prev[i] ) ;
		}
		else
		{
			Kernel_momentum_hidden<<< 1, neurons >>>(  ( (i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i]  , state->d_Out_ff[i] , state->d_W[i+1]  , state->d_Out_ff[i+1] , state->d_delta_val[i]  , 	 state->d_delta_val[i+1] ,  Lrate, iteration , state->dW_prev[i]) ;
		}
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	return total_error ;
}

float trainNNetwork_resilient(NNState *state, float *input,  float *desired , int iteration )
{
	size_t vector_in_bytes =  ((state->weights[0]->width)-1)* (sizeof(float)); // width -1 for bias
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_input_vec, input , vector_in_bytes ,cudaMemcpyHostToDevice));
	size_t target_bytes =  (state->neurons[OUTPUT_LAYER])* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_desired, desired, target_bytes ,cudaMemcpyHostToDevice));

	//2.Feedforward kernel calls  that executed serially
	size_t i, neurons ;
	float total_error=-1 ;
	cudaDeviceSynchronize();
	for ( i = 0 ; i < (state->levels); i++)
	{
		neurons = state->neurons[i] ;

		if ( isPowOf2(state->d_W[i].width)  )
		{
			//optimized kernel - reduction
			Kernel_forward_fast<<< neurons ,state->d_W[i].width , (state->d_W[i].width*sizeof(float)) >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i] , state->d_Out_ff[i]  ) ;
		}
		else
		{
			Kernel_forward<<< 1 , neurons  >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ), state->d_W[i] ,  state->d_Out_ff[i] ) ;
		}
		cudaDeviceSynchronize();
	}

	if ( iteration%(state->sampling) == 0)
	{
		cudaMemcpy( state->ffout[OUTPUT_LAYER] , state->d_Out_ff[OUTPUT_LAYER], state->neurons[OUTPUT_LAYER]*(sizeof(float)) , cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		total_error  = SError_vec(  state->ffout[ OUTPUT_LAYER  ] , desired , state->neurons[ OUTPUT_LAYER ] );
	}
	// 3. Back propagation Kernels call here
	float Lrate = state->lrate ;
	for ( i = OUTPUT_LAYER ;    i != SIZE_MAX    ; i--) //
	{
		neurons = state->neurons[i]   ;
		if (i == OUTPUT_LAYER )
		{
			Kernel_resilient_last<<< 1 , neurons >>>( state->d_Out_ff[i-1] ,  state->d_W[i] , state->d_Out_ff[i] , state->d_desired , state->d_delta_val[i]   , Lrate , state->d_Grad[i] ,state->d_dij_prev[i] , state->dW_prev[i], iteration ) ;
		}
		else
		{
			Kernel_relilient_hidden<<< 1, neurons >>>(  ( (i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i]
									    , state->d_Out_ff[i] , state->d_W[i+1]  , state->d_Out_ff[i+1] , state->d_delta_val[i]  , 	 state->d_delta_val[i+1] ,  Lrate , state->d_Grad[i] , state->d_dij_prev[i] ,  state->dW_prev[i] , iteration	) ;
		}
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	return total_error ;
}

float trainNNetwork_quick(NNState *state, float *input,  float *desired , int iteration )
{
	size_t vector_in_bytes =  ((state->weights[0]->width)-1)* (sizeof(float)); // width -1 for bias
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_input_vec, input , vector_in_bytes ,cudaMemcpyHostToDevice));
	size_t target_bytes =  (state->neurons[OUTPUT_LAYER])* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_desired, desired, target_bytes ,cudaMemcpyHostToDevice));

	//2.Feedforward kernel calls  that executed serially
	size_t i, neurons ;
	float total_error=-1 ;
	cudaDeviceSynchronize();
	for ( i = 0 ; i < (state->levels); i++)
	{
		neurons = state->neurons[i] ;
		if ( isPowOf2(state->d_W[i].width)  )
		{
			//optimized kernel - reduction
			Kernel_forward_fast<<< neurons ,state->d_W[i].width , (state->d_W[i].width*sizeof(float)) >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i] , state->d_Out_ff[i]  ) ;
		}
		else
		{
			Kernel_forward<<< 1 , neurons  >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ), state->d_W[i] ,  state->d_Out_ff[i] ) ;
		}
		cudaDeviceSynchronize();
	}
//copy back results to cpu to calculate error when needed
	if ( iteration%(state->sampling) == 0)
	{
		cudaMemcpy( state->ffout[OUTPUT_LAYER] , state->d_Out_ff[OUTPUT_LAYER], state->neurons[OUTPUT_LAYER]*(sizeof(float)) , cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		total_error  = SError_vec(  state->ffout[ OUTPUT_LAYER  ] , desired , state->neurons[ OUTPUT_LAYER ] );
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3. Back propagation Kernels call here
	for ( i = OUTPUT_LAYER ;    i != SIZE_MAX    ; i--) //
	{
		neurons = state->neurons[i]   ;
		if (i == OUTPUT_LAYER )
		{
			Kernel_quick_last<<< 1 , neurons >>>( state->d_Out_ff[i-1] ,  state->d_W[i] , state->d_Out_ff[i] , state->d_desired , state->d_delta_val[i]   , state->dW_prev[i], state->d_Grad[i] ) ;
		}
		else
		{
			Kernel_quick_hidden<<< 1, neurons >>>(  ( (i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i]
									    , state->d_Out_ff[i] , state->d_W[i+1]  , state->d_Out_ff[i+1] , state->d_delta_val[i]  , 	 state->d_delta_val[i+1] ,state->dW_prev[i], state->d_Grad[i] 	) ;
		}
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	return total_error ;
}

float trainNNetwork_old(NNState *state, float *input, size_t input_len, float *desired  )
{
	//1. copy input vector and desired output for the specific iteration
	size_t vector_in_bytes =  (input_len)* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_input_vec, input , vector_in_bytes ,cudaMemcpyHostToDevice));
	size_t target_bytes =  (state->neurons[OUTPUT_LAYER])* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_desired, desired, target_bytes ,cudaMemcpyHostToDevice));

	//2.Feedforward kernel calls  that executed serially
	size_t vector_out_bytes , sizeW , i, neurons ;
	cudaDeviceSynchronize();
	for ( i = 0 ; i < (state->levels); i++)
	{
		// copy weights to cpu
		sizeW = (state->weights[i]->width )*( state->weights[i]->height)  * sizeof(float);
		CUDA_CHECK_RETURN(cudaMemcpy(state->d_W[i].elements, state->weights[i]->elements, sizeW,cudaMemcpyHostToDevice));
		neurons = state->neurons[i] ;
		vector_out_bytes = (neurons)*(sizeof(float)) ;
		if ( isPowOf2(state->d_W[i].width)  )
		{
			//optimized kernel - reduction
			Kernel_forward_fast<<< neurons ,state->d_W[i].width , (state->d_W[i].width*sizeof(float)) >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i] , state->d_Out_ff[i]  ) ;
		}
		else
		{
			Kernel_forward<<< 1 , neurons  >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ), state->d_W[i] ,  state->d_Out_ff[i] ) ;
		}
		cudaDeviceSynchronize();
		cudaMemcpy( state->ffout[i] , state->d_Out_ff[i], vector_out_bytes , cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3. Back propagation Kernels call here
	float Lrate = state->lrate ;
	for ( i = OUTPUT_LAYER ;    i != SIZE_MAX    ; i--) //
	{

		neurons = state->neurons[i]   ;

		vector_out_bytes = (neurons)*(sizeof(float)) ;
		sizeW = (state->weights[i]->width )*( state->weights[i]->height)  * sizeof(float);//???????????????

		if (i == OUTPUT_LAYER )
		{
			Kernel_back_last<<< 1 , neurons >>>( state->d_Out_ff[i-1] ,  state->d_W[i] , state->d_Out_ff[i] , state->d_desired , state->d_delta_val[i]   , Lrate ) ;
		}
		else
		{

			Kernel_back_hidden<<< 1, neurons >>>(  ( (i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i]
						    , state->d_Out_ff[i] , state->d_W[i+1]  , state->d_Out_ff[i+1] , state->d_delta_val[i]  , 	 state->d_delta_val[i+1] ,  Lrate	) ;
		}
		cudaDeviceSynchronize();
		cudaMemcpy( state->delta_val[i] , state->d_delta_val[i] , vector_out_bytes , cudaMemcpyDeviceToHost);
		cudaMemcpy( state->weights[i]->elements , state->d_W[i].elements , sizeW , cudaMemcpyDeviceToHost);
	}

	float total_error  = SError_vec(  state->ffout[ OUTPUT_LAYER  ] , desired , state->neurons[ OUTPUT_LAYER ] );
	return total_error ;
}

void copyDataToGpu( NNState *state  )
{
	size_t sizeW , i ;
	// copy init random weights to gpu
		for ( i = 0 ; i < (state->levels); i++)
		{
			sizeW = (state->weights[i]->width )*( state->weights[i]->height)  * sizeof(float);
			CUDA_CHECK_RETURN(cudaMemcpy(state->d_W[i].elements, state->weights[i]->elements, sizeW,cudaMemcpyHostToDevice));
			switch ( state->variant )
				{
					case CLASSIC :	break ;
					case MOMENTUM :
					{
						CUDA_CHECK_RETURN(cudaMemcpy(state->dW_prev[i].elements, state->dweight[i]->elements, sizeW,cudaMemcpyHostToDevice));
						break;
					}
					case RESILIENT :
					{
						CUDA_CHECK_RETURN(cudaMemcpy(state->d_Grad[i].elements, state->grad[i]->elements, sizeW,cudaMemcpyHostToDevice));
						CUDA_CHECK_RETURN(cudaMemcpy(state->d_dij_prev[i].elements, state->dij_prev[i]->elements, sizeW,cudaMemcpyHostToDevice));
						CUDA_CHECK_RETURN(cudaMemcpy(state->dW_prev[i].elements, state->dweight[i]->elements, sizeW,cudaMemcpyHostToDevice));
						break;
					}
					case ADAPTIVE :
					{
						CUDA_CHECK_RETURN(cudaMemcpy(state->d_Grad[i].elements, state->grad[i]->elements, sizeW,cudaMemcpyHostToDevice));
						CUDA_CHECK_RETURN(cudaMemcpy(state->d_Lrate[i].elements, state->Lrate[i]->elements, sizeW,cudaMemcpyHostToDevice));
						break;
					}
					case QUICK:
					{
						CUDA_CHECK_RETURN(cudaMemcpy(state->dW_prev[i].elements, state->dweight[i]->elements, sizeW,cudaMemcpyHostToDevice));
						CUDA_CHECK_RETURN(cudaMemcpy(state->d_Grad[i].elements, state->grad[i]->elements, sizeW,cudaMemcpyHostToDevice));
					}


				}

		}
		cudaDeviceSynchronize();
		/*
	// copy multiple-data to gpu

		size_t vector_in_bytes =  ( state->weights[0]->width)* (sizeof(float));// width-1
		for ( i = 0 ; i < (state->maxData); i++)
		{
			CUDA_CHECK_RETURN(cudaMemcpy(state->d_data[i], Data[i*state->weights[0]->width] , vector_in_bytes ,cudaMemcpyHostToDevice));
		}

	//TODO  I also have to copy the desired values !!!
	 * */


}


void copyDataToCpu( NNState *state   )
{
	size_t sizeW , i ;
	size_t vector_out_bytes =  (state->neurons[OUTPUT_LAYER])*(sizeof(float));
	// copy final random weights to Cpu
		for ( i = 0 ; i < (state->levels); i++)
		{
			vector_out_bytes =  (state->neurons[i])*(sizeof(float));
			sizeW = (state->weights[i]->width )*( state->weights[i]->height)  * sizeof(float);
			cudaMemcpy( state->weights[i]->elements , state->d_W[i].elements , sizeW , cudaMemcpyDeviceToHost);
			cudaMemcpy( state->delta_val[i] , state->d_delta_val[i] , vector_out_bytes , cudaMemcpyDeviceToHost);  //debug only!!!!
			cudaMemcpy( state->ffout[i] , state->d_Out_ff[i], state->neurons[i]*(sizeof(float)) , cudaMemcpyDeviceToHost);

			switch (state->variant )
			{
			case MOMENTUM :
			{
				cudaMemcpy( state->dweight[i]->elements , state->dW_prev[i].elements , sizeW , cudaMemcpyDeviceToHost);
				break;
				//matrixPrint_HEADER(state->dweight[i] , 4) ;//debug
			}
			case QUICK :
			{
				cudaMemcpy( state->dweight[i]->elements , state->dW_prev[i].elements , sizeW , cudaMemcpyDeviceToHost);
				cudaMemcpy( state->grad[i]->elements , state->d_Grad[i].elements , sizeW , cudaMemcpyDeviceToHost);
				break;
			}
			case ADAPTIVE :
			{
				cudaMemcpy( state->Lrate[i]->elements , state->d_Lrate[i].elements , sizeW , cudaMemcpyDeviceToHost);
				break;
			}
			case RESILIENT :
			{
				cudaMemcpy( state->grad[i]->elements , state->d_Grad[i].elements , sizeW , cudaMemcpyDeviceToHost);
				cudaMemcpy( state->dweight[i]->elements , state->dW_prev[i].elements , sizeW , cudaMemcpyDeviceToHost);
				cudaMemcpy( state->dij_prev[i]->elements , state->d_dij_prev[i].elements , sizeW , cudaMemcpyDeviceToHost);
				printf("dW \n");
				matrixPrint_HEADER(state->dweight[i] , 4) ;
				printf("Gradient \n");
				matrixPrint_HEADER(state->grad[i] ,4) ;
				printf("dij  \n");
				matrixPrint_HEADER(state->dij_prev[i] , 4) ;
				break;
			}
			case CLASSIC:break;
			}
		}
		cudaDeviceSynchronize();
}

float evalNN(NNState *state, float *input, size_t input_len , int desired   )
{
	size_t vector_in_bytes =  (input_len)* (sizeof(float));
	CUDA_CHECK_RETURN(cudaMemcpy(state->d_input_vec, input , vector_in_bytes ,cudaMemcpyHostToDevice));
	//2.Feedforward kernel calls  that executed serially
	size_t vector_out_bytes , sizeW , i, neurons ;
	cudaDeviceSynchronize();
	for ( i = 0 ; i < (state->levels); i++)
	{
		sizeW = (state->weights[i]->width )*( state->weights[i]->height)  * sizeof(float);
		CUDA_CHECK_RETURN(cudaMemcpy(state->d_W[i].elements, state->weights[i]->elements, sizeW,cudaMemcpyHostToDevice));// auto prepei na ginetai kathe fora
		neurons = state->neurons[i] ;
		vector_out_bytes = (neurons)*(sizeof(float)) ;
		if ( isPowOf2(state->d_W[i].width)  )
		{
			//optimized kernel - reduction
			Kernel_forward_fast<<< neurons ,state->d_W[i].width , (state->d_W[i].width*sizeof(float)) >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ) , state->d_W[i] , state->d_Out_ff[i]  ) ;
		}
		else
		{
			Kernel_forward<<< 1 , neurons  >>>( (	(i==0)? state->d_input_vec : state->d_Out_ff[i-1]  ), state->d_W[i] ,  state->d_Out_ff[i] ) ;
		}

		cudaDeviceSynchronize();
		cudaMemcpy( state->ffout[i] , state->d_Out_ff[i], vector_out_bytes , cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize();


	int pos = vectorMax(  state->ffout[OUTPUT_LAYER] ,state->neurons[OUTPUT_LAYER] ) ;
	// POS IS IN RANGE 0- OUTnEURONS-1

	if ( pos == desired )
	{
		return 1 ;
	}
	else return ( 0 );

}


void freeNNState(NNState *state)
{
	if (state)
	{
		for (size_t i = 0; i < state->levels; i++)
		{
			free(state->weights[i]);
			free(state->ffout[i]);
			free(state->delta_val[i]);
			cudaFree( state->d_Out_ff[i] ) ;
			cudaFree( state->d_delta_val[i] ) ;
			cudaFree(state->d_W[i].elements) ;
			switch (state->variant )
			{
					case MOMENTUM :
					{
						free(state->dweight[i] );
						cudaFree(  state->dW_prev[i].elements) ;
						break;
					}
					case QUICK :
					{

						free(state->dweight[i] );
						cudaFree(  state->dW_prev[i].elements) ;
						free(state->grad[i] );
						cudaFree(  state->d_Grad[i].elements) ;
						break;
					}
					case ADAPTIVE :
					{
						 free(state->Lrate[i] );
						 cudaFree( state->d_Lrate[i].elements) ;
						 break;
					}
					case RESILIENT :
					{
						free(state->dweight[i] );
						cudaFree(  state->dW_prev[i].elements) ;
						free(state->grad[i] );
						cudaFree(  state->d_Grad[i].elements) ;
						free(state->dij_prev[i] );
						cudaFree(  state->d_dij_prev[i].elements) ;
					}
					case CLASSIC:	break;
			}
		}
		cudaFree(state->d_desired);
		cudaFree(state->d_input_vec);
		free(state->neurons);
		free(state->ffout);
		free(state->delta_val);
		free(state);
	}

}


static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	fprintf(stderr, "%s returned %s (%d) at %s:%d\n", statement, cudaGetErrorString(err), err, file, line);
	exit (1);
}

void dbgPrintNNState(NNState *state)
{
	printf("--------Adan NN debug results----------\n--------NN Levels = %zu  ----------------\n" ,  state->levels   );
	printf("---------------------------------------\n");
	size_t neurons ;
	for ( size_t i = 0 ; i < state->levels; i++)
		{
			neurons = state->neurons[i]  ;
			printf("\n Output of Neurons  of Level %zu \n" , (i+1) );
			vectorPrint( state->ffout[i]  , neurons ) ;
			printf("\n Delta Value of each neuron  \n" );
			vectorPrint( state->delta_val[i]  , neurons ) ;
			printf("\n Updated Weights Header :  \n" );
			matrixPrint_HEADER(state->weights[i] , 5 );
			printf("---------------------------------------------------------\n");
		}
}


void dbgPrintNNState2(NNState *state)
{
	printf("--------Adan NN debug results----------\n--------NN Levels = %zu  ----------------\n" ,  state->levels   );
	printf("---------------------------------------\n");

	for ( size_t i = 0 ; i < state->levels; i++)
		{


			printf("\n Updated Weights Header :  \n" );
			matrixPrint_HEADER(state->weights[i] , 5 );
			printf("---------------------------------------------------------\n");
		}
}

void dbgPrintNNState3(NNState *state)
{
	printf("--------Adan NN debug results----------\n--------NN Levels = %zu  ----------------\n" ,  state->levels   );
	printf("---------------------------------------\n");
	vectorPrint(  state->ffout[OUTPUT_LAYER] ,state->neurons[OUTPUT_LAYER] ) ;
	printf("---------------------------------------\n");
}

int NNvar(NNState *state)
{
	int sum=0;
	size_t i ;
	for ( i = 0 ; i < (state->levels); i++)
	{
		sum += (state->weights[i]->width )*( state->weights[i]->height) ;
	}
	return sum ;

}


void showInfo( NNState *state , int instances  )
{
	int w_param = NNvar(state);
	printf("----NN parameters----\n");
	printf("%d instances used for training \n",instances );
	printf("%d weights  \n", w_param );
	printf("Lrate = %1.3f  \n" , state->lrate);
	printf("Max_epochs = %zu  \n" , state->max_iterations);
	printBPA( state->variant );

	for ( size_t i = 0 ; i < state->levels; i++)
	{
		printf("Level %zu with %d neurons   \n" , i+1 , state->neurons[i] );

	}
	printf("-------------------\n");

}
unsigned int isPowOf2(unsigned int x)
{
	// returns 1 when x is power of two
	return !(x & (x - 1));

}
