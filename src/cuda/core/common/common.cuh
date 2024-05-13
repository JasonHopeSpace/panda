
#pragma once

#include <cstdio>
#include <inttypes.h>
#include <sys/time.h>  
#include <cuda_runtime.h>

#include "../config.cuh"

namespace common
{

  #define HANDLE_RESULT_CUDA(statement)                                                               \
    {                                                                                                 \
      if (statement != cudaSuccess)                                                                   \
      {                                                                                               \
          printf("[cuda] Error: handle cuda result error  (line: %u, function: %sfile: %s)\n"         \
          ,  __LINE__, __FUNCTION__, __FILE__);                                                       \
          return statement;                                                                           \
      }                                                                                               \
    }
#if PANDA_ASM
#if PANDA_ASM_32
  typedef unsigned limb_t;
#elif PANDA_ASM_64
  typedef unsigned long long limb_t;
#endif
#else
  typedef unsigned long long limb_t;
#endif

  #define HOST_INLINE __host__ __forceinline__
  #define DEVICE_INLINE __device__ __forceinline__
  #define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  #define HOST_DEVICE __host__ __device__

  #define TIMECONVERTER  1000000  // 1s = 1000 000 us

  cudaError stream_create(cudaStream_t &stream, bool blocking_sync);

  cudaError mem_pool_create(cudaMemPool_t &mem_pool, int device_id);
}
