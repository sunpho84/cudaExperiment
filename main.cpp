#include <cstdio>
#include <unistd.h>

#include <iostream>

#include <cuda_runtime.h>
#include "global.hpp"


using namespace std;

const int N=1<<20;

#include <cuda_runtime.h>

int is_device_pointer_(void* ptr) {
  // Check the attributes of the given pointer.
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  if (err == cudaSuccess) {
    // The pointer is valid. Check if it is a device pointer.
    if (attributes.type == cudaMemoryTypeDevice) {
      // The pointer is a device pointer.
      return 1;
    } else {
      // The pointer is not a device pointer.
      return 2;
    }
  } else {
    // The pointer is not valid.
    return 0;
  }
}

int is_device_pointer(void* ptr) {
  // Try to copy data to or from the given pointer.
  float data = 3.14f;
  cudaError_t err = cudaMemcpy(ptr, &data, sizeof(float), cudaMemcpyHostToDevice);

  if (err == cudaSuccess) {
    // The copy was successful, so the pointer is a valid device pointer.
    return 1;
  } else {
    // The copy failed, so the pointer is not a valid device pointer.
    return 2;
  }
}

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
__device__ float d;

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
				 [a=a.getRef()] __device__(const int i) mutable
  {
    glbA=&a;
  });
  
  printf("waiting for end\n");
  cudaDeviceSynchronize();
  printf("glb ptr: %p, %d\n",glbA,is_device_pointer(glbA));
  
  printf("---------\n");
  
  printf("d: %p, %d\n",&d,is_device_pointer(&d));

  float* dev_ptr;
  cudaError_t err = cudaMalloc(&dev_ptr, N * sizeof(float));
    printf("m: %p, %d\n",dev_ptr,is_device_pointer(dev_ptr));
  cudaFree(dev_ptr);
  
  cudaFree(x);
  cudaFree(y);
}
