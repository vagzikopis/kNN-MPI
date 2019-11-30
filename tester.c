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

  // int nstart=5000  , nend=2500 , nstep=500;                    // corpus
  int dstart=100   , dend=101   , dstep=1;                      // dimensions
  // int kstart=5    , kend=10   , kstep=1;                     // # neighbors
  // int p=4;
  struct timeval start, end;
  int n=2500,k=15,p=4;
  // for(int n=nstart; n<nend; n+=nstep)
  // {
  //   for(int k=kstart; k<kend; k+=kstep)
  //   {
      for(int d=dstart; d<dend; d+=dstep)
      {
        double  * corpus = (double * ) malloc( p*n*d * sizeof(double) );


        for (int i=0;i<p*n*d;i++)
        {
          corpus[i] = ( (double) (rand()) ) / (double) RAND_MAX;
        }

        gettimeofday(&start,NULL);
        knnresult knnres = kNN( corpus, corpus, p*n, p*n, d, k );
        gettimeofday(&end,NULL);
        printf("np:%d d:%d k:%d time:%lf\n",n*p,d,k, (double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec));
        int isValidC = validateResult( knnres, corpus, corpus, p*n, p*n, d, k, COLMAJOR );
        // int isValidR = validateResult( knnres, corpus, query, p*n, p*m, d, k, ROWMAJOR );
        if(isValidC)
        {
          printf("Tester validation: "GRN"%s NEIGHBORS\n"RESET, STR_CORRECT_WRONG[isValidC]);
        }else
        {
          printf("Tester validation: "RED"%s NEIGHBORS\n"RESET, STR_CORRECT_WRONG[isValidC]);
        }

        free( corpus );
      }
  //   }
  // }
  return 0;

}
