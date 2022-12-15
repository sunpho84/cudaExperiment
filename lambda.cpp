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

enum class StorLoc{ON_HOST,ON_DEVICE};

template <template <StorLoc> typename T>
struct ManualUnifier
{
  T<StorLoc::ON_HOST> _hostData;
  
  T<StorLoc::ON_DEVICE> _deviceData;
  
  __device__ T<StorLoc::ON_DEVICE>& _data(std::bool_constant<true>)
  {
    return _deviceData;
  }
  
  __host__ T<StorLoc::ON_HOST>& _data(std::bool_constant<false>)
  {
    return _hostData;
  }
  
  __host__ __device__ decltype(auto) data()
  {
    if constexpr(CompilingForDevice)
      return _deviceData;
    else
      return _hostData;
  }
  
  __host__ void sync_host_with_device()
  {
    _hostData=_deviceData;
  }
  
  __host__ void sync_device_with_host()
  {
    _deviceData=_hostData;
  }
  
  bool allocated;
  
  void alloc(const int& ext_n)
  {
    _hostData.alloc(ext_n);
    _deviceData.alloc(ext_n);
    
    allocated=true;
  }
  
  void dealloc()
  {
    _hostData.dealloc();
    _deviceData.dealloc();
    
    allocated=false;
  }
  
  ManualUnifier()
  {
    allocated=false;
  }
  
  ~ManualUnifier()
  {
    if(allocated)
      dealloc();
  }
};

template <StorLoc SL>
struct DataHolder
{
  static constexpr StorLoc sL=SL;
  
  int n;
  
  int* data;
  
  void alloc(const int& ext_n)
  {
    n=ext_n;
    if constexpr(SL==StorLoc::ON_HOST)
      data=new int[n];
    else
      cudaMalloc(&data,sizeof(int)*n);
  }
  
  void dealloc()
  {
    if constexpr(SL==StorLoc::ON_HOST)
      delete[] data;
    else
      cudaFree(data);
    
    data=nullptr;
    n=0;
  }
  
  template <StorLoc SOURCE_SL>
  DataHolder& operator=(const DataHolder<SOURCE_SL>& oth)
  {
    static constexpr StorLoc DEST_SL=SL;
    
    constexpr cudaMemcpyKind kind=
      (SOURCE_SL==StorLoc::ON_HOST)
		?((DEST_SL==StorLoc::ON_HOST)?
		  cudaMemcpyHostToHost:
		  cudaMemcpyHostToDevice)
		:((DEST_SL==StorLoc::ON_HOST)?
		  cudaMemcpyDeviceToHost:
		  cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(data,oth.data,sizeof(int)*n,kind);
    
    return *this;
  }
};

void testManual()
{
  ManualUnifier<DataHolder> manual;
  
  manual.alloc(10);
  
  for(int i=0;i<10;i++)
    manual.data().data[i]=i;
  
  manual.sync_device_with_host();
  
  ManualUnifier<DataHolder> manual2;
  manual2.alloc(10);
  manual2._deviceData=manual._deviceData;
  manual2.sync_host_with_device();
  
  for(int i=0;i<10;i++)
    printf("%d %d \n",manual.data().data[i],manual2.data().data[i]);
  
  // constexpr StorLoc SL=std::remove_pointer<decltype(manual.data())>::type::sL;
  // //std::integral_constant<StorLoc,SL>& a=1;
  // static_assert((SL==StorLoc::ON_DEVICE) xor (not CompilingForDevice),"ma come 1");
  // //static_assert(SL==StorLoc::ON_HOST or not CompilingForDevice,"ma come 2");
}


// __device__ int _aaa_device;

// int _aaa_host;

// __device__ int& _aaa(std::bool_constant<true>)
//   {
//     return _aaa_device;
//   }
  
// __host__ int& _aaa(std::bool_constant<false>)
//   {
//     return _aaa_host;
//   }

// __host__ __device__ int& aaa()
// {
//   return _aaa(std::bool_constant<CompilingForDevice>());
// }

  
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

// template <typename T>
// struct Ref
// {
//   int* data;
  
//   int size;
  
//   Ref(int* data,const int& size) : data(data),size(size)
//   {
//   }
// };

// template <StorLoc SL>
// struct Tens
// {
//   int* data;
  
//   int size;
  
//   __host__ Ref<Tens> GetRef() const
//   {
//     return Ref<Tens>(data,size);
//   }
  
//   Tens() : data(nullptr),size(0)
//   {
//   }
  
//   Tens(const int size) : data(new int[size]),size(size)
//   {
//   }
  
//   ~Tens()
//   {
//     delete[] data;
//     size=0;
//   }
  
//   explicit Tens(const Tens&) =delete;
  
//   __host__ Tens(const Ref<Tens>& ref) : Tens(size)
//   {
//     memcpy(data,ref.data,sizeof(int)*size);
//   }
// };

template <typename T>
struct Ex
{
  T& s;
  
  Ex(T& s) : s(s)
  {
  }
};

struct S
{
  explicit S(const S&) = delete;
  // {
  // }
  
  S()
  {
  }
};

void t(Ex<S> s)
{
}

void testNoCopiable()
{
  int e=1;;

  {
    //int& _e=e;
    int e=e;
    printf("%d %d\n",)
  }
  
  const int min=0;
  const int max=2;
  const int length=max-min;
  const int nthreads=128;
  
  const dim3 block_dimension(nthreads);
  const dim3 grid_dimension((length+block_dimension.x-1)/block_dimension.x);
  
  // const Tens<StorLoc::ON_DEVICE> b;
  // auto _b=b.GetRef();
  // aaa()=1;
  
  S s;
  Ex<S> exs(s);
  t(s);
  cuda_generic_kernel<<<block_dimension,grid_dimension>>>(min,max,
							  [=] __device__(const int& index) mutable
							  {
							    if(index<max)
							      {
								// aaa()=1;
								exs;
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
  testManual();
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



