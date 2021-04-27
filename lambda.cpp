#include <iostream>
#include <stdio.h>

template <typename...Args>
void decript_cuda_error(const cudaError_t& rc,Args&&...args)
{
  if(rc!=cudaSuccess)
    {
      std::cerr<<"cuda raised error: "<<cudaGetErrorString(rc)<<std::endl;
      ((std::cerr << args << '\n'), ...);
      exit(1);
    }
}


[[ maybe_unused ]]
constexpr bool CompilingForDevice=
#ifdef __CUDA_ARCH__
  true
#else
  false
#endif
  ;

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

enum class StorLoc{ON_HOST,ON_DEVICE};

template <typename T>
struct Ref
{
  int* data;
  
  int size;
  
  Ref(int* data,const int& size) : data(data),size(size)
  {
  }
};

template <StorLoc SL>
struct A
{
  int* data;
  
  int size;
  
  __host__ Ref<A> GetRef() const
  {
    return Ref<A>(data,size);
  }
  
  A() : data(nullptr),size(0)
  {
  }
  
  A(const int size) : data(new int[size]),size(size)
  {
  }
  
  ~A()
  {
    delete[] data;
    size=0;
  }
  
  explicit A(const A&) =delete;
  
  __host__ A(const Ref<A>& ref) : A(size)
  {
    memcpy(data,ref.data,sizeof(int)*size);
  }
};

void testNoCopiable()
{
  const int min=0;
  const int max=2;
  const int length=max-min;
  const int nthreads=128;
  
  const dim3 block_dimension(nthreads);
  const dim3 grid_dimension((length+block_dimension.x-1)/block_dimension.x);
  
  const A<StorLoc::ON_DEVICE> b;
  auto _b=b.GetRef();
  cuda_generic_kernel<<<block_dimension,grid_dimension>>>(min,max,
							  [=] __device__(const int& index) mutable
							  {
							    if(index<max)
							      {
								_b;
							      }
							  });
  cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////

// struct Encaps
// {
//   int var;
//   int *ptr;
// };

// __managed__ Encaps* enc;

//using Double=double[2];

int main()
{
  // // std::cout<<a.ptr<<std::endl;
  
  // Aa bb;
  // std::cout<<bb.ptr<<std::endl;
  // int nDevices;
  //   if(cudaGetDeviceCount(&nDevices)!=cudaSuccess)
  //     {
  // 	std::cerr<<"no CUDA enabled device"<<std::endl;
  // 	exit(1);
  //     }
    
  //   printf("Number of CUDA enabled devices: %d\n",nDevices);
  //   for(int i=0;i<nDevices;i++)
  //     {
  // 	cudaDeviceProp deviceProp;
  // 	cudaGetDeviceProperties(&deviceProp,i);
  // 	printf(" CUDA Enabled device %d/%d: %d.%d\n",i,nDevices,deviceProp.major,deviceProp.minor);
  //     }
    
  //   //assumes that if we are seeing multiple gpus, there are nDevices ranks to attach to each of it
  //   decript_cuda_error(cudaSetDevice(0),"Unable to set device 0");
  // // cudaMemcpy(&gpu::a,&a,sizeof(int),cudaMemcpyHostToDevice);
  //   double* glb;

  //   cudaMalloc((void**)&glb,2*sizeof(double));
    
  
  // double locs[2]={0,78};
  // decript_cuda_error(cudaMemcpy(locs,glb,2*sizeof(double),cudaMemcpyDeviceToHost));
  // printf("%lg %lg\n",locs[0],locs[1]);
  
  // cudaFree(glb);
  
  // //int a=locVolh();
  
  return 0;
}


/////////////////////////////////////////////////////////////////

// __device__ void encaps_test()
// {
//   enc->var=1;
// }


// int _locVolhHostVar;

//     /// Half the local volume
// int& _locVolhHost()
// {
//   return _locVolhHostVar;
// }

// __device__ int _locVolhDeviceVar;

//     /// Half the local volume
// __device__ int& _locVolhDevice()
// {
//   return _locVolhDeviceVar;
// }

// __host__ __device__ const int& locVolh()
//     {
// #ifdef __CUDA_ARCH__
//       return _locVolhDevice();
// #else
//       return _locVolhHost();
// #endif
//     }


