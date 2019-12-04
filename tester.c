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

int main(int argc, char *argv[])
{
  if(argc < 3 )
  {
    printf("Please enter 3 arguments.\nn(corpus per process size)\nd(dimensions)\nk(nearest neighbors)\n");
  }
  int n = atoi(argv[1]);
  int d = atoi(argv[2]);
  int k = atoi(argv[3]);
  struct timeval start, end;
  double  * corpus = (double * ) malloc( n*d * sizeof(double) );

  for (int i=0;i<n*d;i++)
  {
    corpus[i] = ( (double) (rand()) ) / (double) RAND_MAX;
  }

  gettimeofday(&start,NULL);
  knnresult knnres = kNN( corpus, corpus, n, n, d, k );
  gettimeofday(&end,NULL);
  printf("n=%d, d=%d, k=%d\nExecution Time:%lf\n",n,d,k,(double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec));
  int isValidC = validateResult( knnres, corpus, corpus, n, n, d, k, COLMAJOR );
  if(isValidC)
  {
    printf("Tester validation: "GRN"%s NEIGHBORS\n"RESET, STR_CORRECT_WRONG[isValidC]);
  }else
  {
    printf("Tester validation: "RED"%s NEIGHBORS\n"RESET, STR_CORRECT_WRONG[isValidC]);
  }

  free( corpus );
  return 0;

}
