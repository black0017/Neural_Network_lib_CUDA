#include <AdalNN.h>
#include "read_file.h"
#include <time.h>


#define ARRAYLEN(x)  (sizeof(x)/sizeof(x[0]))

#define CLASSES 4
#define VALUE 0.7
#define VALOW -0.7
int main(int argc, char **argv)
{
	time_t START;
	START = time(NULL);
	int row = 0 , max_cols = 11 ,max_rows ;
	float B[25011][10] ;
	int T[25011];
	size_t  p ;
	int max_epo= 100000, input_vec_len = 10  , sampling =100 ;
	float min_error = 100 , percentage= 0 , error ;
	float const learnRate = 0.05 , threshold=0.001 ;
	printf("starting . . . .\n") ;
	FILE *nn_train_data = fopen("pokertraining.txt", "r");
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
	fclose(nn_train_data);
	edit_array(&B[0][0] ,input_vec_len , max_rows ) ;
	//print_arrayHeader( &B[0][0] ,input_vec_len , max_rows);
	NNState *neuralNet1 = initNNState( input_vec_len,  learnRate , max_epo , sampling , CLASSIC  , 2 , 100, CLASSES ) ;
	showInfo(neuralNet1 ,max_rows  );
	copyDataToGpu( neuralNet1 ) ;//weights
	//dbgPrintNNState(neuralNet1);
// training
	int idx ;
	for (i =0 ; i<max_epo ; i++)
	{
		p = rand()%max_rows ;
		idx =( T[p]<= (CLASSES-1) )? T[p]  : (CLASSES-1) ; //IDX IS IN RANGE 0  -  CLASSES-1
		target[ idx ] = VALUE;//desired class
		//printf("T[p] = %d  \n" , T[p]   );
		//vectorPrint(target , CLASSES);

		error = trainNNetwork( neuralNet1, &B[p][0], target , i );
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
	dbgPrintNNState(neuralNet1);
	printf("Done training at iteration %d  \nMin_error = %f || current_error = %f\n " ,i,  min_error , error ) ;
	printf("\n-------VaLidation Test with training examples for AdalNN --------------------------------\n" ) ;
	for (size_t i =0 ; i<max_rows ; i++)
	{
		idx =( T[i]<= (CLASSES-1) )? T[i]  : (CLASSES-1) ;
		percentage += evalNN( neuralNet1, &B[i][0], input_vec_len , idx);
	}
	printf( "\n%1.f correct out of %d \n  " , percentage , max_rows  ) ;

	percentage = (percentage/max_rows)*100;
	printf( "\n%f success percentage\n  " , percentage  ) ;

	time_t STOP;
	STOP = time(NULL);
	printf("TOTAL TIME = %ld SECONDS\n", STOP-START );
	return 0;
}



/*
 *
 *
 *
 *
 *
 * 	 float C[1918][10];//B sub matrix
	 int D[1918];// T sub vector
	 int idc=0 , idx  ;
	 for (size_t i =0 ; i<max_rows ; i++)
	 {
		 if ( T[i]>=2 )
		 {
			 idx =T[i]-2 ;// range 0-7
			 D[idc] = idx  ; // range 0-7
			 for ( size_t j =0 ; j<10 ; j++ )
			 {
				 C[idc][j]  = B[i][j] ;

			 }
			 idc++ ;
		 }
	 }
	// printf("max_rows NN2 = %d\n" , idc ) ;
	 //vectorPrintINT_HEADER(D ,idc);
 *
	float target2[8] ;
		for (int u=0 ; u<8 ; u++)
		{
			target2[u]=-0.9 ;
		}
	NNState *neuralNet2 = initNNState( input_vec_len,  learnRate , max_epo , 2 ,40,8) ;
	showInfo(neuralNet2 ,idc  );
	int i ,q  ;
		for (i =0 ; i<max_epo ; i++)
		{
			p = rand()%idc ;
			q =D[p];// range 0-7
			target2[ q ] = 0.9;//desired class
			error = trainNNetwork_resilient( neuralNet2, &C[p][0], input_vec_len, target2 , i );
			//error = trainNNetwork( neuralNet2, &C[p][0], input_vec_len, target2 , i );
			target2[ idx  ] = -0.9 ;

			if ((i%40)==0)
			{
				if( error < min_error )
				{
					min_error  = error ;
					printf("iteration %d  , error=%f\n", i , min_error );
					if (min_error <= 0.041 )
						{
						printf("------stoppedddd1111 ----------\n" ) ;
						printf("------stoppedddd1111 ----------\n" ) ;
						break ;
						}
				}

			}
		}


		printf("-------VaLidation Test with training examples for AdalNN --------------------------------\n" ) ;
		int desired ;
		for (size_t i =0 ; i<idc  ; i++)
		{
			desired  =D[i] ;// range 0-7
			percentage += evalNN( neuralNet2, &C[i][0], input_vec_len , desired);
		}
		percentage = (percentage/idc)*100;
		printf( "\n %f percentage\n  " , percentage   ) ;

	freeNNState(neuralNet2);
*/
