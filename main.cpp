#include <cstdio>
#include "global.hpp"

#include <iostream>

using namespace std;

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

  template <typename IMin,
	    typename IMax,
	    typename F>
  __global__
  void cuda_generic_kernel(const IMin min,
			   const IMax max,
			   F f)
  {
    const auto i=min+blockIdx.x*blockDim.x+threadIdx.x;
    if(i<max)
      f(i);
  }

struct A
{
  int* const count;
  
  A() :
    count(new int(1))
  {
    printf("creating A at %p, count(%p): %d\n",this,count,*count);
  }
  
  A(const A& oth) :
    count(oth.count)
  {
    (*count)++;
    printf("copying A from %p to %p, count(%p): %d\n",&oth,this,count,*count);
  }
  
  __host__ __device__
  ~A()
  {
    (*count)--;
    printf("destroying A at %p, count(%p): %d\n",this,count,*count);
    if((*count)==0) delete count;
  }
};

int main()
{
  cout<<"ecco"<<endl;
  cout<<u<<endl;

  cudaMallocManaged(&x,N*sizeof(double));
  cudaMallocManaged(&y,N*sizeof(double));
  
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N);
  cudaDeviceSynchronize();
  
  printf("---------\n");

  A a;
  
  cuda_generic_kernel<<<10,10>>>(0,100,
				 [a] __device__(const int i)
  {
  });
  cudaDeviceSynchronize();
  
  printf("---------\n");
  
  
  cudaFree(x);
  cudaFree(y);
}
