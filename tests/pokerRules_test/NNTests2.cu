#include <AdalNN.h>
#include "read_file.h"
#include <time.h>
#define ARRAYLEN(x)  (sizeof(x)/sizeof(x[0]))
#define CLASSES 5
#define VALUE 0.8
#define VALOW -0.8
//*************************************************
int main(int argc, char **argv)
{
	time_t START;
	START = time(NULL);
	int row = 0 , max_cols = 11 ,max_rows ;
	float B[25011][10] ;
	int T[25011];
	size_t  p ;
	int max_epo= 50000, input_vec_len = 10  , sampling =100 ;
	float min_error = 100 , percentage= 0 , error ;
	float const learnRate = 0.03 , threshold=0.0008 ;
	printf("starting . . . .\n") ;
	FILE *nn_train_data = fopen("poker_random.txt", "r");
	if (!nn_train_data)
		return 1;

	float target[ CLASSES ] ;
	for (int u=0 ; u<CLASSES ; u++)
	{
		target[u]= VALOW;
	}

	char *next_word = NULL;
	int i = 0;
	float curr_float;

	while ((next_word = readNextWordFromFile(nn_train_data))) {

		curr_float = (float)atof(next_word);
		if (i==input_vec_len)
		{
			T[row]= (int)curr_float ;
		}

		if (i!=input_vec_len)
		{
			 B[row][i] = curr_float ;

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
	int trainset = max_rows*0.9 ;
	fclose(nn_train_data);
	//preprossecing
	edit_array(&B[0][0] ,input_vec_len , max_rows ) ;

	NNState *neuralNet1 = initNNState( input_vec_len,  learnRate , max_epo , sampling , CLASSIC  , 2 , 128,  CLASSES ) ;
	showInfo(neuralNet1 ,max_rows  );

	copyDataToGpu( neuralNet1 ) ;
// training
	int idx ;
	for (i =0 ; i<max_epo ; i++)
	{
		p = rand()% trainset  ;
		idx =( T[p]<= (CLASSES-1) )? T[p]  : (CLASSES-1) ; //IDX IS IN RANGE 0  -  CLASSES-1

		target[ idx ] = VALUE;//desired class

		//error = trainNNetwork( neuralNet1, &B[p][0], target , i, NUM_OF_DATA );
		error = trainNNetwork_test( neuralNet1, &B[p][0], target , i );
		//error = trainNNetwork_momentum( neuralNet1, &B[p][0], target , i );
		//error = trainNNetwork_resilient( neuralNet1, &B[p][0],  target , i ); //ready?
		//error = trainNNetwork_quick( neuralNet1, &B[p][0], target , i );

		target[ idx  ] = VALOW ;
		if (!(i%neuralNet1->sampling)  )
		{
			if( error < min_error )
			{
				min_error  = error ;
				printf("iteration %d  , error=%f\n", i , min_error );
				if (min_error <=  threshold)
					{
					printf("------stopped----------\n" ) ;
					break ;
					}
			}
		}
	}
	copyDataToCpu(neuralNet1) ;
	//dbgPrintNNState(neuralNet1);
	printf("Done training at iteration %d  \nMin_error = %f || current_error = %f\n " ,i,  min_error , error ) ;
	printf("\n-------VaLidation Test with training examples for AdalNN --------------------------------\n" ) ;
	for (size_t i =trainset ; i<max_rows ; i++)
	{
		idx =( T[i]<= (CLASSES-1) )? T[i]  : (CLASSES-1) ;
		percentage += evalNN( neuralNet1, &B[i][0], input_vec_len , idx);
	}
	printf( "\n%1.f correct out of %d \n  " , percentage , max_rows-trainset  ) ;

	percentage = (percentage/(max_rows-trainset))*100;
	printf( "\n%f success percentage\n  " , percentage  ) ;

	time_t STOP;
	STOP = time(NULL);
	printf("TOTAL TIME = %ld SECONDS\n", STOP-START );
	return 0;
}
