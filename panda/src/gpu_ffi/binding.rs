use super::*;

extern "C" {
    pub fn panda_get_device_number(count: *mut ::std::os::raw::c_int) -> PandaError;

    pub fn panda_get_device(device_id: *mut ::std::os::raw::c_int) -> PandaError;

    pub fn panda_set_device(device_id: ::std::os::raw::c_int) -> PandaError;

    pub fn panda_stream_create(stream: *mut PandaStream, blocking_sync: bool) -> PandaError;

    pub fn panda_stream_wait_event(stream: PandaStream, event: PandaEvent) -> PandaError;

    pub fn panda_stream_synchronize(stream: PandaStream) -> PandaError;

    pub fn panda_stream_query(stream: PandaStream) -> PandaError;

    pub fn panda_stream_destroy(stream: PandaStream) -> PandaError;

    pub fn panda_launch_host_fn(
        stream: PandaStream,
        fn_: PandaHostFn,
        user_data: *mut ::std::os::raw::c_void,
    ) -> PandaError;

    pub fn panda_event_create(
        event: *mut PandaEvent,
        blocking_sync: bool,
        disable_timing: bool,
    ) -> PandaError;

    pub fn panda_event_record(event: PandaEvent, stream: PandaStream) -> PandaError;

    pub fn panda_event_sync(event: PandaEvent) -> PandaError;

    pub fn panda_event_query(event: PandaEvent) -> PandaError;

    pub fn panda_event_destroy(event: PandaEvent) -> PandaError;

    pub fn panda_mem_get_info(free: *mut SizeT, total: *mut SizeT) -> PandaError;

    pub fn panda_malloc(ptr: *mut *mut ::std::os::raw::c_void, size: SizeT) -> PandaError;

    pub fn panda_malloc_host(ptr: *mut *mut ::std::os::raw::c_void, size: SizeT) -> PandaError;

    pub fn panda_free(ptr: *mut ::std::os::raw::c_void) -> PandaError;

    pub fn panda_free_host(ptr: *mut ::std::os::raw::c_void) -> PandaError;

    pub fn panda_host_register(ptr: *mut ::std::os::raw::c_void, size: SizeT) -> PandaError;

    pub fn panda_host_unregister(ptr: *mut ::std::os::raw::c_void) -> PandaError;

    pub fn panda_device_disable_peer_access(device_id: ::std::os::raw::c_int) -> PandaError;

    pub fn panda_device_enable_peer_access(device_id: ::std::os::raw::c_int) -> PandaError;

    pub fn panda_memcpy(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: SizeT,
    ) -> PandaError;

    pub fn panda_memcpy_async(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: SizeT,
        stream: PandaStream,
    ) -> PandaError;

    pub fn panda_memset(
        ptr: *mut ::std::os::raw::c_void,
        value: ::std::os::raw::c_int,
        count: SizeT,
    ) -> PandaError;

    pub fn panda_memset_async(
        ptr: *mut ::std::os::raw::c_void,
        value: ::std::os::raw::c_int,
        count: SizeT,
        stream: PandaStream,
    ) -> PandaError;

    pub fn panda_mem_pool_create(
        pool: *mut PandaMemPool,
        device_id: ::std::os::raw::c_int,
    ) -> PandaError;

    pub fn panda_mem_pool_destroy(pool: PandaMemPool) -> PandaError;

    pub fn panda_malloc_from_pool_async(
        ptr: *mut *mut ::std::os::raw::c_void,
        size: SizeT,
        pool: PandaMemPool,
        stream: PandaStream,
    ) -> PandaError;

    pub fn panda_free_async(ptr: *mut ::std::os::raw::c_void, stream: PandaStream) -> PandaError;

    pub fn panda_msm_setup_bn254() -> PandaError;

    pub fn panda_msm_execute_bn254(configuration: MSMConfiguration) -> PandaError;

    pub fn panda_msm_execute_bn254_host(configuration: MSMConfiguration) -> PandaError;

    pub fn panda_msm_tear_down() -> PandaError;

    pub fn panda_ntt_setup_bn254(ptr: *mut ::std::os::raw::c_void) -> PandaError;

    pub fn panda_ntt_execute_bn254(configuration: NTTConfiguration) -> PandaError;

    pub fn panda_ntt_execute_bn254_v1(configuration: NttconfigurationV1) -> PandaError;

    pub fn panda_ntt_tear_down() -> PandaError;
}
