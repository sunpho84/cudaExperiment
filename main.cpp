#include "global.hpp"

const int N=1<<20;



int main()
{
  cudaMallocManaged(&x,N*sizeof(double));
  cudaMallocManaged(&y,N*sizeof(double));
  
  cudaFree(x);
  cudaFree(y);
}
