use super::*;
use std::ffi::c_void;
use std::ptr::addr_of_mut;

#[derive(Clone, Debug)]
pub enum PandaGpuError {
    GetDeviceCountError,
    SetDeviceError,
    DeviceGetDeviceMemoryInfoError,

    CreateContextError,
    InitUnitTypeError,
    MSMBasesAddrError,
    NTTOmegaAddrError,

    SetBasesErr,
    SchedulingErr,
    GetExponentAddressErr,
    GetResultAddressesErr,
    StartProcessingErr,
    FinishProcessingErr,
    DestroyContextErr,
    BasesIndexErr,

    MemPoolCreateErr,
    AsyncPoolMallocErr,
    AsyncMemcopyErr,
    NttExecErr,
    StremCreateErr,
    StreamDestroyErr,
    StreamWaitEventErr,
    StreamSyncErr,

    EventCreateErr,
    EventRecordErr,
    EventDestroyErr,
    EventSyncErr,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PandaStream {
    pub handle: *mut ::std::os::raw::c_void,
}

impl PandaStream {
    pub fn new() -> Result<PandaStream, PandaGpuError> {
        let mut new = Self::null();
        if unsafe { panda_stream_create(new.as_mut_ptr(), true) } != 0 {
            return Err(PandaGpuError::StremCreateErr);
        };

        Ok(new)
    }
    pub fn destroy(self) -> Result<(), PandaGpuError> {
        if unsafe { panda_stream_destroy(self) } != 0 {
            return Err(PandaGpuError::StreamDestroyErr);
        }

        Ok(())
    }

    pub fn wait(self, event: PandaEvent) -> Result<(), PandaGpuError> {
        if unsafe { panda_stream_wait_event(self, event) } != 0 {
            return Err(PandaGpuError::StreamWaitEventErr);
        }

        Ok(())
    }

    pub fn sync(self) -> Result<(), PandaGpuError> {
        if unsafe { panda_stream_synchronize(self) } != 0 {
            return Err(PandaGpuError::StreamSyncErr);
        }
        Ok(())
    }

    pub fn null() -> PandaStream {
        PandaStream {
            handle: std::ptr::null_mut() as *mut c_void,
        }
    }

    fn as_mut_ptr(&mut self) -> *mut PandaStream {
        addr_of_mut!(*self)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PandaEvent {
    pub handle: *mut ::std::os::raw::c_void,
}

impl PandaEvent {
    pub fn new() -> Result<PandaEvent, PandaGpuError> {
        let mut event = PandaEvent::null();
        if unsafe { panda_event_create(addr_of_mut!(event), true, true) } != 0 {
            return Err(PandaGpuError::EventCreateErr);
        }
        Ok(event)
    }

    pub fn record(self, stream: PandaStream) -> Result<(), PandaGpuError> {
        if unsafe { panda_event_record(self, stream) } != 0 {
            return Err(PandaGpuError::EventRecordErr);
        }

        Ok(())
    }
    pub fn destroy(self) -> Result<(), PandaGpuError> {
        if unsafe { panda_event_destroy(self) } != 0 {
            return Err(PandaGpuError::EventDestroyErr);
        }

        Ok(())
    }

    pub fn null() -> PandaEvent {
        PandaEvent {
            handle: std::ptr::null_mut() as *mut c_void,
        }
    }

    pub fn sync(self) -> Result<(), PandaGpuError> {
        if unsafe { panda_event_sync(self) } != 0 {
            return Err(PandaGpuError::EventSyncErr);
        }

        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PandaMemPool {
    pub handle: *mut ::std::os::raw::c_void,
}

impl PandaMemPool {
    pub fn new(device_id: usize) -> Result<PandaMemPool, PandaGpuError> {
        let mut mem_pool = Self::null();
        let result =
            unsafe { panda_mem_pool_create(addr_of_mut!(mem_pool), device_id as i32) } == 0;
        if !result {
            return Err(PandaGpuError::MemPoolCreateErr);
        }

        Ok(mem_pool)
    }

    pub fn null() -> PandaMemPool {
        PandaMemPool {
            handle: std::ptr::null_mut() as *mut c_void,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PandaDeviceInfo {
    pub free: u64,
    pub total: u64,
}

pub type PandaHostFn =
    ::std::option::Option<unsafe extern "C" fn(user_data: *mut ::std::os::raw::c_void)>;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum PandaMSMResultCoordinateType {
    Jacobian = 0,
    Projective,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MSMConfiguration {
    pub mem_pool: PandaMemPool,
    pub stream: PandaStream,
    pub bases: *mut ::std::os::raw::c_void,
    pub scalars: *mut ::std::os::raw::c_void,
    pub results: *mut ::std::os::raw::c_void,
    pub log_scalars_count: ::std::os::raw::c_uint,
    pub msm_result_coordinate_type: PandaMSMResultCoordinateType,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NTTConfiguration {
    pub mem_pool: PandaMemPool,
    pub stream: PandaStream,
    pub d_src: *mut ::std::os::raw::c_void,
    pub d_dst: *mut ::std::os::raw::c_void,
    pub log_n: ::std::os::raw::c_uint,
    pub flag: *mut ::std::os::raw::c_uint,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NttconfigurationV1 {
    pub mem_pool: PandaMemPool,
    pub stream: PandaStream,
    pub d_src: *mut ::std::os::raw::c_void,
    pub d_dst: *mut ::std::os::raw::c_void,
    pub omega: *const ::std::os::raw::c_void,
    pub log_n: ::std::os::raw::c_uint,
    pub flag: *mut ::std::os::raw::c_uint,
}
