
#include <cuda_runtime_api.h>
#include "common/common.cuh"
#include "panda_interface.cuh"
#include "curve/bn254/config.cuh"
#include "unit/msm/msm_cuda.cuh"
#include "unit/msm/msm_host.cuh"
#include "unit/ntt/fft.cuh"


panda_error panda_get_device_number(int *count)
{
  return static_cast<panda_error>(cudaGetDeviceCount(count));
}

panda_error panda_get_device(int *device_id)
{
  return static_cast<panda_error>(cudaGetDevice(device_id));
}

panda_error panda_set_device(int device_id)
{
  return static_cast<panda_error>(cudaSetDevice(device_id));
}

panda_error panda_stream_create(panda_stream *stream, bool blocking_sync)
{
  return static_cast<panda_error>(common::stream_create(reinterpret_cast<cudaStream_t &>(stream->handle), blocking_sync));
}

panda_error panda_stream_wait_event(panda_stream stream, panda_event event)
{
  return static_cast<panda_error>(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream.handle), static_cast<cudaEvent_t>(event.handle)));
}

panda_error panda_stream_sync(panda_stream stream)
{
  return static_cast<panda_error>(cudaStreamSynchronize(static_cast<cudaStream_t>(stream.handle)));
}

panda_error panda_stream_destroy(panda_stream stream)
{
  return static_cast<panda_error>(cudaStreamDestroy(static_cast<cudaStream_t>(stream.handle)));
}

panda_error panda_launch_host_fn(panda_stream stream, panda_host_fn fn, void *user_data)
{
  return static_cast<panda_error>(cudaLaunchHostFunc(static_cast<cudaStream_t>(stream.handle), fn, user_data));
}

panda_error panda_event_create(panda_event *event, bool blocking_sync, bool disable_timing)
{
  int flags = (blocking_sync ? cudaEventBlockingSync : cudaEventDefault) | (disable_timing ? cudaEventDisableTiming : cudaEventDefault);
  return static_cast<panda_error>(cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(&(event->handle)), flags));
}

panda_error panda_event_record(panda_event event, panda_stream stream)
{
  return static_cast<panda_error>(cudaEventRecord(static_cast<cudaEvent_t>(event.handle), static_cast<cudaStream_t>(stream.handle)));
}

panda_error panda_event_sync(panda_event event)
{
  return static_cast<panda_error>(cudaEventSynchronize(static_cast<cudaEvent_t>(event.handle)));
}

panda_error panda_event_query(panda_event event)
{
  return static_cast<panda_error>(cudaEventQuery(static_cast<cudaEvent_t>(event.handle)));
}

panda_error panda_event_destroy(panda_event event)
{
  return static_cast<panda_error>(cudaEventDestroy(static_cast<cudaEvent_t>(event.handle)));
}

panda_error panda_mem_get_info(size_t *free, size_t *total)
{
  return static_cast<panda_error>(cudaMemGetInfo(free, total));
}

panda_error panda_malloc(void **ptr, size_t size)
{
  return static_cast<panda_error>(cudaMalloc(ptr, size));
}

panda_error panda_malloc_host(void **ptr, size_t size)
{
  return static_cast<panda_error>(cudaMallocHost(ptr, size));
}

panda_error panda_free(void *ptr)
{
  return static_cast<panda_error>(cudaFree(ptr));
}

panda_error panda_free_host(void *ptr)
{
  return static_cast<panda_error>(cudaFreeHost(ptr));
  }

panda_error panda_host_register(void *ptr, size_t size)
{
  return static_cast<panda_error>(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
}

panda_error panda_host_unregister(void *ptr)
{
  return static_cast<panda_error>(cudaHostUnregister(ptr));
}

panda_error panda_memcpy(void *dst, const void *src, size_t count)
{
  return static_cast<panda_error>(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
}

panda_error panda_memcpy_async(void *dst, const void *src, size_t count, panda_stream stream)
{
  return static_cast<panda_error>(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, static_cast<cudaStream_t>(stream.handle)));
}

panda_error panda_memset(void *ptr, int value, size_t count)
{
  return static_cast<panda_error>(cudaMemset(ptr, value, count));
}

panda_error panda_memset_async(void *ptr, int value, size_t count, panda_stream stream)
{
  return static_cast<panda_error>(cudaMemsetAsync(ptr, value, count, static_cast<cudaStream_t>(stream.handle)));
}

panda_error panda_mem_pool_create(panda_mem_pool *pool, int device_id)
{
  return static_cast<panda_error>(common::mem_pool_create(reinterpret_cast<cudaMemPool_t &>(pool->handle), device_id));
}

panda_error panda_mem_pool_destroy(panda_mem_pool pool)
{
  return static_cast<panda_error>(cudaMemPoolDestroy(reinterpret_cast<cudaMemPool_t>(pool.handle)));
}

panda_error panda_malloc_from_pool_async(void **ptr, size_t size, panda_mem_pool pool, panda_stream stream)
{
  return static_cast<panda_error>(cudaMallocFromPoolAsync(ptr, size, static_cast<cudaMemPool_t>(pool.handle), static_cast<cudaStream_t>(stream.handle)));
}

panda_error panda_free_async(void *ptr, panda_stream stream)
{
  return static_cast<panda_error>(cudaFreeAsync(ptr, static_cast<cudaStream_t>(stream.handle)));
}

panda_error panda_msm_setup_bn254()
{
  return static_cast<panda_error>(panda_msm_v1::core_msm_setup_bn254());
}

panda_error panda_msm_execute_bn254(const panda_msm_configuration exec_cfg)
{
  return static_cast<panda_error>(panda_msm_v1::core_msm_execute_bn254(exec_cfg));
}

panda_error panda_msm_execute_bn254_host(const panda_msm_configuration exec_cfg)
{
  return static_cast<panda_error>(panda_msm::core_msm_execute_bn254_host(exec_cfg));
}

panda_error panda_msm_tear_down()
{
  return static_cast<panda_error>(panda_msm_v1::core_msm_tear_down());
}

panda_error panda_ntt_setup_bn254(void* input_omega)
{

  return static_cast<panda_error>(panda_ntt::core_ntt_setup_bn254(input_omega));
}

panda_error panda_ntt_execute_bn254(const panda_ntt_configuration exec_cfg)
{
  return static_cast<panda_error>(panda_ntt::core_ntt_execute_bn254(exec_cfg));
}

panda_error panda_ntt_execute_bn254_v1(const panda_ntt_configuration_v1 exec_cfg)
{
  return static_cast<panda_error>(panda_ntt::core_ntt_execute_bn254_v1(exec_cfg));
}

panda_error panda_ntt_tear_down()
{
  return static_cast<panda_error>(panda_ntt::core_ntt_tear_down());
}
