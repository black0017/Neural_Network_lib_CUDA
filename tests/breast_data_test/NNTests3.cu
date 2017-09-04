#include <AdalNN.h>
#include "read_file.h"
#include <time.h>

#define ARRAYLEN(x)  (sizeof(x)/sizeof(x[0]))

#define CLASSES 2

int main(int argc, char **argv)
{
	time_t START;
	START = time(NULL);
	int row = 0 , max_cols = 32 ,max_rows ;
	float B[600][30] ;
	char T[600];
	size_t  p ;
	int max_epo= 15000, input_vec_len = 30 , sampling=1;
	float min_error = 100 , percentage= 0 , error ,  learnRate = 0.001 , threshold = 0.0001;
	printf("starting . . . .\n") ;
	FILE *nn_train_data = fopen("breast_randomized.txt", "r");
	float target1[] = { 1 , -1 } ;
	float target2[] = { -1 , 1 } ;

	 if (nn_train_data)
	 {
	        char *next_word = NULL;
	        int i = 0;
	        float curr_float;
	        while ((next_word = readNextWordFromFile(nn_train_data)))
	        {
	            if (i==0)
	            {
	            	curr_float = (double)atof(next_word);
	            	//Id[ row ] = curr_float ;
	            }
	            if (i==1)
	            {
	            	T[row]= next_word[0] ;
	            	//if (next_word[0] == 'M') printf("M\n");
	            	//if (next_word[0] == 'B') printf("B\n");
	            }
	            if ( i >=2   )
	            {
	            	curr_float = (float)atof(next_word);
	            	//if (row<=20)  printf("float = %f \n",  curr_float );
	            	B[row][i-2] =  curr_float ;
	            }

	            free(next_word);
	            i++;
	            if (i >= max_cols)
	            {
	                i = 0;
	                row ++ ;
	            }
	        }
	        max_rows=row ;
	        fclose(nn_train_data);
	    }
	 //print_arrayHeader( &B[0][0] ,input_vec_len , max_rows);
	 edit_array(&B[0][0] ,input_vec_len , max_rows ) ;
	 print_arrayHeader( &B[0][0] ,input_vec_len , max_rows);
//NNState *initNNState( int input_vec_len, float Lrate ,size_t max_iter , int sampling  , BPA variant ,  size_t levels, ... );
	NNState *neuralNet1 = initNNState( input_vec_len,  learnRate , max_epo , sampling ,  CLASSIC , 2 , 12 ,  CLASSES ) ;
	showInfo(neuralNet1 ,max_rows  );
	copyDataToGpu( neuralNet1 ) ;//weights
// training
	int i ;
	int trainset =  max_rows*0.5;
	for (i =0 ; i<max_epo ; i++)
	{
		p = rand()%trainset ;

		switch ( T[p] )
				{
				case 'M':
					//error = trainNNetwork_old( neuralNet1, &B[p][0], input_vec_len, target1 );
					//error = trainNNetwork( neuralNet1, &B[p][0], target1 , i );
					//error = trainNNetwork_MOM( neuralNet1, &B[p][0], target1 , i );
					//error = trainNNetwork_resilient( neuralNet1, &B[p][0],  target1, i );
					error = trainNNetwork_test( neuralNet1, &B[p][0], target1 , i );
					//error = trainNNetwork_quick( neuralNet1, &B[p][0], target1 , i );
					break;
				case 'B':
					//error = trainNNetwork_old( neuralNet1, &B[p][0], input_vec_len, target2  );
					//error = trainNNetwork( neuralNet1, &B[p][0], target2 , i );
					//error = trainNNetwork_MOM( neuralNet1, &B[p][0], target2 , i );
					//error = trainNNetwork_resilient( neuralNet1, &B[p][0],  target2 , i );
					error = trainNNetwork_test( neuralNet1, &B[p][0], target2 , i );
					//error = trainNNetwork_quick( neuralNet1, &B[p][0], target2 , i );
					break;
				}

		if (!(i%neuralNet1->sampling)  )
		{
			if( error < min_error )
			{
				min_error  = error ;
				printf("iteration %d  , error=%f\n", i , min_error );
				if (min_error <= threshold )
					{
					printf("------stoppedddd1111 ----------\n" ) ;
					printf("------stoppedddd1111 ----------\n" ) ;
					break ;
					}
			}

		}




	}
	copyDataToCpu(neuralNet1) ;//weights
	printf("Done training at iteration %d  \nMin_error = %f ||||current_error = %f\n " ,i,  min_error , error ) ;
	//dbgPrintNNState(neuralNet1);
	printf("-------VaLidation Test with training examples for AdalNN --------------------------------\n" ) ;
	for (size_t i =trainset ; i<max_rows ; i++)
	{
		switch ( T[i] )
					{
					case 'M':
						percentage += evalNN( neuralNet1, &B[i][0], input_vec_len , 0 );
						break;
					case 'B':
						percentage += evalNN( neuralNet1, &B[i][0], input_vec_len , 1 );
						break;
					}
	}
	int num = percentage ;
	percentage = (percentage/(max_rows-trainset))*100;
	printf( "\n%f percentage\n num= %d out of %d \n " , percentage , num  , (max_rows-trainset) ) ;
	freeNNState(neuralNet1);
	time_t STOP;
	STOP = time(NULL);
	printf("TOTAL TIME = %ld SECONDS\n", STOP-START );
// */
	return 0;
}
