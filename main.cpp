#include <cstdio>
#include <unistd.h>
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

struct A;

struct F;

__managed__ A* glbA;

struct F
{
  int nRef;
  
  F() :
    nRef(0)
  {
    printf("Creating F\n");
  }
  
  ~F()
  {
    printf("Destroying F\n");
  }
  
  void action()
  {
    printf("action!\n");
  }
  
  A getRef();
};

struct A
{
  F& f;
  
  int& nRef;
  
  A(F& f) :
    f(f),
    nRef(f.nRef)
  {
    nRef++;
    printf("creating A at %p, nRef(%p): %d\n",this,&nRef,nRef);
  }
  
  A(const A& oth) :
    f(oth.f),
    nRef(oth.nRef)
  {
    nRef++;
    printf("copying A from %p to %p, nRef(%p): %d\n",&oth,this,&nRef,nRef);
  }
  
  __host__ __device__
  ~A()
  {
    nRef--;
    printf("destroying A at %p, nRef(%p): %d\n",this,&nRef,nRef);
#ifndef __CUDA_ARCH__
    if(nRef==0) f.action();
#endif
  }
};

A F::getRef()
{
  return *this;
}

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

  F a;
  
  cuda_generic_kernel<<<10,10>>>(0,100,
				 [a=a.getRef()] __device__(const int i)
  {
    glbA=&a;
  });
  
  printf("waiting for end\n");
  cudaDeviceSynchronize();
  printf("glb ptr: %p\n",glbA);
  
  printf("---------\n");
  
  
  cudaFree(x);
  cudaFree(y);
}
