#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "../inc/knnring.h"

knnresult kNN(double *X, double *Y, int n, int m,int d, int k)
{
  knnresult result;
  if(k<=0 || k>n)
  {
    printf("\033[31m" "\t\t\tK must be >0 and <=n\n" "\033[0m" );
    return result;
  }
  struct timeval start,end;
  double* D = calloc(n*m,sizeof(double));
  knnresult *result_ptr;

  result_ptr = &result;
  result.m = m;
  result.k = k;
  result_ptr->nidx = malloc((m*k)*sizeof(int));
  result_ptr->ndist = calloc((m*k),sizeof(double));
  gettimeofday(&start,NULL);
  calcDistance(X,Y,n,m,d,D);
  gettimeofday(&end,NULL);
  // printf("Distance time:%lf\n",(double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec));
  double *tempCol = malloc(n*sizeof(double));
  int *tempIdx = malloc(n*sizeof(int));
  int index;
  int zeros = 0;
  for(int i=0; i<m; i++)
  {
    for(int j=0; j<n; j++)
    {
      tempCol[j] = D[j + n*i];
      tempIdx[j] = j;
    }
    result_ptr->ndist[m*(k-1)+i] = quickselect(tempCol,0,n-1,k,tempIdx,&index);
    result_ptr->nidx[m*(k-1)+i] = tempIdx[index];
    for(int j=k-2; j>=0; j--)
    {
      result_ptr->ndist[m*j+i] = quickselect(tempCol,0,j,j+1,tempIdx,&index);
      result_ptr->nidx[m*j+i] = tempIdx[index];
    }
  }

  free(D);
  free(tempCol);
  return result;
}

void calcDistance(double *X, double *Y, int n, int m, int d, double* D)
{
  double alpha = -2.0;
  double beta = 0.0;
  double* powX = calloc(n,sizeof(double));
  double* powY = calloc(m,sizeof(double));

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, d, alpha, X, n, Y, m, beta, D, n);
  for(int i=0; i<n; i++)
  {
    for(int j=0; j<d; j++)
    {
      powX[i] += pow(*(X+n*j+i), 2);
    }
  }

  for(int i=0; i<m; i++)
  {
    for(int j=0; j<d; j++)
    {
      powY[i] += pow(*(Y+m*j+i), 2);
    }
  }
  for(int i=0; i<n; i++)
  {
    for(int j=0; j<m; j++)
    {
      *(D+n*j+i) =  *(D+n*j+i) + powX[i] + powY[j];
      if(*(D+n*j+i) < 0.000001)
      {
        *(D+n*j+i) = 0.0;
      }
      *(D+n*j+i) = sqrt(*(D+n*j+i));
    }
  }
  free(powX);
  free(powY);
}

void swap_1(double* a, double* b)
{
    double t = *a;
    *a = *b;
    *b = t;
}

void swap_2(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(double* arr, int l, int r, int* idx)
{
    double x = *(arr + r);
    int i = l;
    for (int j = l; j <= r - 1; j++)
    {
        if (*(arr + j) <= x)
        {
            swap_1(&*(arr + i), &*(arr + j));
            swap_2(&*(idx + i), &*(idx + j));
            i++;
        }
    }
    swap_1(&*(arr + i), &*(arr + r));
    swap_2(&*(idx + i), &*(idx + r));
    return i;
}

double quickselect(double* arr, int l, int r, int k, int* idx, int* return_idx)
{
    if (k > 0 && k <= r - l + 1)
    {
        int index = partition(arr, l, r,idx);

        if (index - l == k - 1)
        {
          *return_idx = index;
          return *(arr + index);
        }

        if (index - l > k - 1)
            return quickselect(arr, l, index - 1, k,idx,return_idx);

        return quickselect(arr, index + 1, r, k - index + l - 1,idx,return_idx);
    }

    return 0;
}
