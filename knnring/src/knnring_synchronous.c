#define RED   "\x1B[31m"
#define RESET "\x1B[0m"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cblas.h>
#include <time.h>
#include "../inc/knnring.h"
//! Compute distributed all-kNN of points in X, using blocking MPI communication
/*!
\param X Data points [n-by-d]
\param n Number of data points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
knnresult distrAllkNN(double * X, int n, int d, int k)
{
  double *Y = malloc(n*d*sizeof(double));
  knnresult result;
  knnresult temp_result;
  MPI_Status Stat;
  int p,id,tag=1,dst,rcv;
  MPI_Comm_rank(MPI_COMM_WORLD, &id); // process ID
  MPI_Comm_size(MPI_COMM_WORLD, &p); // # tasks
  double *X_copy = malloc(n*d*sizeof(double));
  memcpy(Y,X,n*d*sizeof(double));
  double waittime=0;
  struct timeval start, end;
  double result_distances[k],temp_distances[k],finalD[k];
  int result_indexes[k],temp_indexes[k],finalI[k];
  int counter = 1;
  for(int i=0; i<p; i++)
  {
    //If i==0 calculate kNN and skip result update //
    if(i==0)
    {
      result = kNN(X,Y,n,n,d,k);
      for(int i=0; i<n; i++)
      {
        for(int j=0; j<k; j++)
        {
          if(id != 0)
          {
            result.nidx[n*j+i] += (id-1)*n ;
          }else
          {
            result.nidx[n*j+i] += (p-1)*n ;
          }
        }
      }
    }else
    {
      // Calculate kNN and update your local result //
      temp_result = kNN(X,Y,n,n,d,k);
      for(int j=0; j<n; j++)
      {
        for(int w=0; w<k; w++)
        {
          if(id == 0)
          {
            temp_result.nidx[n*w+j]+= (p-i-1)*n;
          }else if(id-i-1 >= 0)
          {
            temp_result.nidx[n*w+j]+= (id-i-1)*n;
          }else
          {
            temp_result.nidx[n*w+j]+= (p-counter)*n;
          }
          result_distances[w] = result.ndist[n*w+j];
          result_indexes[w] = result.nidx[n*w+j];
          temp_distances[w] = temp_result.ndist[n*w+j];
          temp_indexes[w] = temp_result.nidx[n*w+j];
        }
        mergeArrays(result_distances,temp_distances,result_indexes,temp_indexes,k,k,finalD,finalI);
        for(int w=0; w<k; w++)
        {
          result.ndist[n*w+j] = finalD[w];
          result.nidx[n*w+j] = finalI[w];
        }
      }
      if(id-i-1<0) counter++;
    }
    // If this is the last iteration skip communication with other processes //
    if (i!=p-1)
    {
      gettimeofday(&start,NULL);
      // Determine sender - receiver for each process based on their process id //
      if(id%2 == 0)
      {
        dst = id+1;
        if(id==p-1)
        {
          dst = 0;
        }
        rcv = id-1;
        if(id==0)
        {
          rcv=p-1;
        }
        //              Blocking communication                //
        //  Even processes first send and then receive corpus //
        MPI_Send(X, n*d, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
        MPI_Recv(X, n*d, MPI_DOUBLE, rcv, tag, MPI_COMM_WORLD, &Stat);
      }else{
        dst = id+1;
        if(id==p-1)
        {
          dst = 0;
        }
        rcv = id-1;
        memcpy(X_copy,X,n*d*sizeof(double));
        //              Blocking communication                //
        //  Odd processes first receive and then send their corpus //
        MPI_Recv(X, n*d, MPI_DOUBLE, rcv, tag, MPI_COMM_WORLD, &Stat);
        MPI_Send(X_copy, n*d, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
      }
      gettimeofday(&end,NULL);
      waittime += (double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec);
    }
  }
  if(id==0)
  {
    printf("Approximate Time lost on communications:"RED"%lf\n"RESET, waittime);
  }

  free(X_copy);
  free(Y);
  return result;
}

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!
\param X Corpus data points [n-by-d]
\param Y Query data points [m-by-d]
\param n Number of data points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
knnresult kNN(double *X, double *Y, int n, int m,int d, int k)
{
  knnresult result;
  // Check for valid parameters //
  if(k<=0 || k>n)
  {
    printf("\033[31m" "\t\t\tK must be >0 and <=n\n" "\033[0m" );
    return result;
  }
  double* D = calloc(n*m,sizeof(double));
  knnresult *result_ptr;

  result_ptr = &result;
  result.m = m;
  result.k = k;
  result_ptr->nidx = malloc((m*k)*sizeof(int));
  result_ptr->ndist = calloc((m*k),sizeof(double));
  // Calculate distances between points //
  // and store the result in a MxN array //
  calcDistance(X,Y,n,m,d,D);

  // printf("Distance time:%lf\n",(double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec));
  double *tempCol = malloc(n*sizeof(double));
  int *tempIdx = malloc(n*sizeof(int));
  int index;
  int zeros = 0;
  for(int i=0; i<m; i++)
  {
    for(int j=0; j<n; j++)
    {
      // Temporary store distances of //
      // all points(Y) from all points(X) //
      tempCol[j] = D[j + n*i];
      tempIdx[j] = j;
    }
    // Select the k-nearest neighbours for all points(Y) //
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

// This function calculates distances between m-points stored in //
// Y from n-points stored in X. The result is stored in array D, //
// with size MxN.                                               //
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
// Swap double values //
void swap_1(double* a, double* b)
{
    double t = *a;
    *a = *b;
    *b = t;
}
// Swap int values //
void swap_2(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}
// Partition used for quickselect //
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

// Implement k-select and return the value of the k-selected item     //
// Also, the index of the k-selected is stored in return_idx variable //
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

void mergeArrays(double arr1[], double arr2[], int indx1[], int indx2[], int n1, int n2, double arr3[], int indx3[])
{
    int i = 0, j = 0, k = 0;
    // Traverse both array
    while (i<n1 && j <n2)
    {
        // Check if current element of first
        // array is smaller than current element
        // of second array. If yes, store first
        // array element and increment first array
        // index. Otherwise do same with second array
        if (arr1[i] < arr2[j])
        {
          arr3[k] = arr1[i];
          indx3[k++] = indx1[i++];
        }
        else
        {
          arr3[k] = arr2[j];
          indx3[k++] = indx2[j++];
        }
    }
}
