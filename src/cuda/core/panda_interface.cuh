#pragma once


#include "curve/bn254/config.cuh"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum panda_error
{
  panda_success = 0,           
  panda_error_invalid_value = 1,
  panda_error_memory_allocation = 2,
  panda_error_not_ready = 600
} panda_error;

typedef struct panda_stream
{
  void *handle;
} panda_stream;

typedef struct panda_event
{
  void *handle;
} panda_event;

typedef struct panda_mem_pool
{
  void *handle;
} panda_mem_pool;

typedef enum panda_msm_result_coordinate_type
{
    JACOBIAN = 0,
    PROJECTIVE,
} panda_msm_result_coordinate_type;

typedef void (*panda_host_fn)(void *user_data);
panda_error panda_get_device_number(int *count);
panda_error panda_get_device(int *device_id);
panda_error panda_set_device(int device_id);
panda_error panda_stream_create(panda_stream *stream, bool blocking_sync);
panda_error panda_stream_wait_event(panda_stream stream, panda_event event);
panda_error panda_stream_sync(panda_stream stream);
panda_error panda_stream_query(panda_stream stream);
panda_error panda_stream_destroy(panda_stream stream);
panda_error panda_launch_host_fn(panda_stream stream, panda_host_fn fn, void *user_data);
panda_error panda_event_create(panda_event *event, bool blocking_sync, bool disable_timing);
panda_error panda_event_record(panda_event event, panda_stream stream);
panda_error panda_event_sync(panda_event event);
panda_error panda_event_query(panda_event event);
panda_error panda_event_destroy(panda_event event);
panda_error panda_mem_get_info(size_t *free, size_t *total);
panda_error panda_malloc(void **ptr, size_t size);
panda_error panda_malloc_host(void **ptr, size_t size);
panda_error panda_free(void *ptr);
panda_error panda_free_host(void *ptr);
panda_error panda_host_register(void *ptr, size_t size);
panda_error panda_host_unregister(void *ptr);
panda_error panda_memcpy(void *dst, const void *src, size_t count);
panda_error panda_memcpy_async(void *dst, const void *src, size_t count, panda_stream stream);
panda_error panda_memset(void *ptr, int value, size_t count);
panda_error panda_memset_async(void *ptr, int value, size_t count, panda_stream stream);
panda_error panda_mem_pool_create(panda_mem_pool *pool, int device_id);
panda_error panda_mem_pool_destroy(panda_mem_pool pool);
panda_error panda_malloc_from_pool_async(void **ptr, size_t size, panda_mem_pool pool, panda_stream stream);
panda_error panda_free_async(void *ptr, panda_stream stream);

typedef struct panda_msm_configuration
{
  panda_mem_pool mem_pool;
  panda_stream stream;
  void *bases;
  void *scalars;
  void *results;
  unsigned log_scalars_count;
  panda_msm_result_coordinate_type msm_result_coordinate_type;
} msm_configuration;

panda_error panda_msm_setup_bn254();
panda_error panda_msm_execute_bn254(const panda_msm_configuration exec_cfg);
panda_error panda_msm_execute_bn254_host(const panda_msm_configuration exec_cfg);
panda_error panda_msm_tear_down();

typedef struct panda_ntt_configuration
{
  panda_mem_pool mem_pool;                  // The memory pool that will be used for temporary allocations needed by the execution
  panda_stream stream;                      // The stream on which the execution will be scheduled
  void *d_src;                          // Pointer to the inputs of this execution, can be either pinned or pageable host memory or device memory pointer.
  void *d_dst;                         // Pointer to the outputs of this execution, can be either pinned or pageable host memory or device memory pointer.
  unsigned log_n;                       // Log2 of the number of values
  void *flag;
} ntt_configuration;

typedef struct panda_ntt_configuration_v1
{
  panda_mem_pool mem_pool;                  // The memory pool that will be used for temporary allocations needed by the execution
  panda_stream stream;                      // The stream on which the execution will be scheduled
  void *d_src;                          // Pointer to the inputs of this execution, can be either pinned or pageable host memory or device memory pointer.
  void *d_dst;                         // Pointer to the outputs of this execution, can be either pinned or pageable host memory or device memory pointer.
  void *d_omega;
  unsigned log_n;                       // Log2 of the number of values
  void *flag;
} ntt_configuration_v1;

panda_error panda_ntt_setup_bn254(void* input_omega);
panda_error panda_ntt_execute_bn254(panda_ntt_configuration exec_cfg);
panda_error panda_ntt_tear_down();
panda_error panda_ntt_execute_bn254_v1(const panda_ntt_configuration_v1 exec_cfg);

#ifdef __cplusplus
} // extern "C"
#endif
