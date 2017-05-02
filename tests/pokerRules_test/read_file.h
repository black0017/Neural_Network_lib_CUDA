#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>



char *readNextWordFromFile(FILE *fd)
{
    //constant value to use as a standard allocation size
    static size_t const alloc_size = 127;

    //buffer to store the word that was read. The buffer's
    //size is dynamically allocated (and reallocated if neccessary)
    //to fit the total characters of each word
    char *buff = NULL;

    //points to the current index in the buff array
    size_t idx = 0;

    //stores the current size of the buff array
    size_t curr_size = 0;

    //temporary int to store the current character read from the file
    int c;

    //skip all initial whitespace until we hit a word character or EOF
    while ( (c = fgetc(fd)) != EOF && isspace(c));

    //if we haven't hit EOF and we hit a valid word character instead
    if (!feof(fd)) {

        //first we allocate size equal to alloc_size for our buff array
        curr_size = alloc_size;
        buff = (char*)malloc(curr_size);
        assert(buff != NULL);

        //loop through each character of the word, until we hit whitespace or EOF
        //this is a do/while loop so that we won't skip the last character that
        //was read in the previous while() loop
        do {

            //if the current index is bigger than the current size
            //it means that we have read more characters than they could fit into
            //the array
            if (idx >= curr_size) {

                //so we reallocate the array into a bigger size
                curr_size += alloc_size;
                buff = (char*)realloc(buff, curr_size);
                assert(buff != NULL);

            }

            //store the character to the array and increment our index
            buff[idx++] = (char)c;

        } while ((c = fgetc(fd)) != EOF && !isspace(c));

        //in case we read as many characters as the array can fit,
        //we need to reallocate, to fit the null-terminating
        //character into the array
        if (idx >= curr_size) {

            buff = (char*)realloc(buff, curr_size + 1);
            assert(buff != NULL);

        }

        buff[idx] = '\0';

    }
    return buff;
}


void edit_array( float *A, size_t width, size_t height)
{
// 1.find the mean of coloums
//2. calculate the standard deviation of coloums
// 3.find new normalized value of (A[i][j] - μ) / σ
	size_t register i , j ;
	float std;
	float sum, mean , var , max;
	// traversing rows
	for( j = 0; j < width; ++j)
    {
	  sum=0; mean=0; var=0 , max=-1;
	  //1 find the mean values of cols
	  for( i = 0; i < height; ++i)
	  {
		  sum = sum + A[i * width + j] ;
		  if ( A[i * width + j] > max )
		  {

			  max = A[i * width + j] ;
		  }

	  }


  	  mean= sum/(height) ;
  	//2. calculate the standard deviation of coloums
  	  for( i = 0; i < height; ++i)
  		  var = var + (mean-A[i * width + j] )*(mean-A[i * width + j] ) ;

  	  var = var/(height-1);
  	  std=(sqrt(var));
  	// 3.find new normalized value of (x - μ) / σ
  	  for( i = 0; i < height; ++i)
  	  {
  		  A[i * width + j] = (A[i * width + j]-mean)/std ;
  		  //A[i * width + j] = A[i * width + j]/max ;
  	  }
    }
	return ;

}

