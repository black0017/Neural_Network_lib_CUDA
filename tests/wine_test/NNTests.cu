#include <AdalNN.h>
#include "read_file.h"
#include <time.h>

#define ARRAYLEN(x)  (sizeof(x)/sizeof(x[0]))

int main(int argc, char **argv)
{
	time_t START;
	START = time(NULL);
	int row = 0 , max_cols = 14 ,max_rows = 178 ;
	float B[178][13] ;
	int T[178];
	//float target1[]={1,0,0}, target2[]={0,1,0} ,target3[]={0,0,1} ;
	float target1[]={1,-1,-1}, target2[]={-1,1,-1} ,target3[]={-1,-1,1} ;
	size_t  p ;
	int max_epo= 100, input_vec_len = 13;
	float min_error = 100 , percentage= 0 , error ,  learnRate= 0.01  ;


	FILE *nn_data_file = fopen("wine.data", "r");

	 if (nn_data_file)
	 {
	        char *next_word = NULL;
	        int i = 0;
	        float curr_float;

	        while ((next_word = readNextWordFromFile(nn_data_file))) {
	            curr_float = (float)atof(next_word);
	            if (i==0)
	            {
	            	T[row]= (int)curr_float ;
	            }

	            if (i!=0)
	            {
	            	 B[row][i-1] = curr_float ;
	            }

	            free(next_word);
	            i++;
	            if (i >= max_cols)
	            {
	                i = 0;
	                row ++ ;
	            }
	        }
	        fclose(nn_data_file);
	    }
	 edit_array(&B[0][0] ,input_vec_len , max_rows ) ;

	 //print_arrayHeader( &B[0][0] ,max_cols-1 , max_rows);
	 //print_array( &B[0][0] ,max_cols-1 , max_rows );
	 //vectorPrintINT(T,max_rows);

	 // B is the input vector
	 // T is the target output
	 //NNState *initNNState( int input_vec_len, float  Lrate ,  size_t levels, ... );
	 // LAST LEVEL NEURONS MUST BE AS THE SIZE OF THE TARGET VECTOR
	 // input vector must be the size of variable input_vec_len
	 // target vector must be the size of last level

	 //print all parameters
	NNState *neuralNet = initNNState( input_vec_len,  learnRate , 10000 , 10 , RESILIENT ,  2 ,10, 3 ) ;
	printf("numOfWeights = %d  \n", NNvar(neuralNet ) ) ;
	showInfo( neuralNet , max_rows );
	copyDataToGpu( neuralNet ) ;//weights
	for (int i =0 ; i<max_epo ; i++)
	{
		p = rand()%max_rows ;
		switch ( (int)T[p] )
		{
		case 1:
			//error = trainNNetwork( neuralNet, &B[p][0], target1 , i );
			error = trainNNetwork_resilient( neuralNet, &B[p][0],  target1 , i );
			break;
		case 2:
			//error = trainNNetwork( neuralNet, &B[p][0], target2 , i );
			error = trainNNetwork_resilient( neuralNet, &B[p][0],  target2 , i );
			break;
		case 3:
			//error = trainNNetwork( neuralNet, &B[p][0], target3 , i );
			error = trainNNetwork_resilient( neuralNet, &B[p][0],  target3 , i );
			break;
		}
		if ((i%200)==0)
		{
			if( error < min_error )
			{
				min_error  = error ;
				printf("iteration %d  , error=%f\n", i , min_error );
				if (min_error <= 0.01)
					{
					printf("------stoppedddd1111 ----------\n" ) ;
					printf("------stoppedddd1111 ----------\n" ) ;
					break ;
					}
			}
		}
	}
	printf("Done training ! min_error = %f \t currentERROR = %f \n" , min_error  , error) ;
	copyDataToCpu( neuralNet ) ;
	//dbgPrintNNState(neuralNet);
	printf("-------Evaluation Test for AdalNN --------------------------------\n" ) ;

	int idx;
	for (size_t i =0 ; i<max_rows ; i++)
	{
		idx=T[i]-1 ; //range 0 -2
		percentage += evalNN( neuralNet, &B[i][0], input_vec_len , idx);
	}
	percentage = (percentage/max_rows)*100;
	printf( "\n ------------------ \n  "   ) ;
	printf( "\n %f percentage\n  " , percentage   ) ;
	freeNNState(neuralNet);
	time_t STOP;
	STOP = time(NULL);
	printf("TIME = %ld SECONDS\n", STOP-START );
	return 0;
}
