#ifndef KNN_H
#define KNN_H

// Definition of the kNN result struct
typedef struct knnresult{
int *nidx;//!< Indices (0-based) of nearest neighbors [m-by-k]
double *ndist;//!< Distance of nearest neighbors [m-by-k]
int m;//!< Number of query points [scalar]
int k;//!< Number of nearest neighbors [scalar]
} knnresult;

knnresult distrAllkNN(double * X, int n, int d, int k);
void mergeArrays(double arr1[], double arr2[], int indx1[], int indx2[], int n1, int n2, double arr3[], int indx3[]);
void calcDistance(double *X, double *Y, int n, int m, int d,double *D);
double quickselect(double* arr, int l, int r, int k, int* idx, int* index);
int partition(double* arr, int l, int r, int *idx);
void swap_1(double* a, double* b);
void swap_2(int* a, int* b);
int searchIndex(double *D, int n, int m_point, double distance,int *index,int k);

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!
\param X Corpus data points [n-by-d]
\param Y Query data points [m-by-d]
\param n Number of data points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

//! Compute distributed all-kNN of points in X
/*!
\param X Data points [n-by-d]
\param n Number of data points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
#endif
