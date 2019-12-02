/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/
#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include "knnring.h"

#include "tester_helper.h"

int main()
{
  FILE *fp = fopen("sequentialdata.txt", "a");
  fprintf(fp, "n,d,k,time\n");
  struct timeval start, end;
  int nstart=500  , nend=25500 , nstep=500;                    // corpus
  int dstart=10   , dend=31   , dstep=10;                      // dimensions
  int kstart=10   , kend=51   , kstep=10;                     // # neighbors
  for(int n=nstart; n<nend; n+=nstep)
  {
    for(int k=kstart; k<kend; k+=kstep)
    {
      for(int d=dstart; d<dend; d+=dstep)
      {
        double  * corpus = (double * ) malloc( n*d * sizeof(double) );


        for (int i=0;i<n*d;i++)
        {
          corpus[i] = ( (double) (rand()) ) / (double) RAND_MAX;
        }

        gettimeofday(&start,NULL);
        knnresult knnres = kNN( corpus, corpus, n, n, d, k );
        gettimeofday(&end,NULL);
        fprintf(fp,"%d,%d,%d,%lf\n",n,d,k,(double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec));
        // int isValidC = validateResult( knnres, corpus, corpus, p*n, p*n, d, k, COLMAJOR );
        // // int isValidR = validateResult( knnres, corpus, query, p*n, p*m, d, k, ROWMAJOR );
        // if(isValidC)
        // {
        //   printf("Tester validation: "GRN"%s NEIGHBORS\n"RESET, STR_CORRECT_WRONG[isValidC]);
        // }else
        // {
        //   printf("Tester validation: "RED"%s NEIGHBORS\n"RESET, STR_CORRECT_WRONG[isValidC]);
        // }

        free( corpus );
      }
    }
  }
  fclose(fp);
  return 0;

}
