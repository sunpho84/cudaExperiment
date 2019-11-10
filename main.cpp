#include "global.hpp"

const int N=1<<20;

__host__ __device__
void add(double* x,double* y,int i)
{
      y[i] = x[i] + y[i];
}

__global__
void add(int n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    add(x,y,i);
}

int main()
{
  cudaMallocManaged(&x,N*sizeof(double));
  cudaMallocManaged(&y,N*sizeof(double));
  
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N);
  
  cudaDeviceSynchronize();
  
  cudaFree(x);
  cudaFree(y);
}
