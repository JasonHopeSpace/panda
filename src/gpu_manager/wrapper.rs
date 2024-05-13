use std::ffi::c_void;
use std::fmt;

use super::*;
use ark_std::{end_timer, start_timer};

#[derive(Clone, Debug)]
pub struct PandaGpuManager {
    device_id: usize,
    mem_pool: PandaMemPool,
    default_stream: PandaStream,
    h2d_stream: PandaStream,
    d2h_stream: PandaStream,
    exec_stream: PandaStream,
    pub d_bases: Vec<*mut c_void>,
    pub d_scalars: Vec<*mut c_void>,
    pub scalars_len: Vec<usize>, // bytes count
    msm_result_coordinate_type: PandaMSMResultCoordinateType,
}
unsafe impl Send for PandaGpuManager {}
unsafe impl Sync for PandaGpuManager {}

#[derive(Clone, Debug)]
pub enum PandaGpuManagerInitUnitType {
    PandaGpuManagerInitUnitTypeNone,
    PandaGpuManagerInitUnitTypeMSM,
    PandaGpuManagerInitUnitTypeNTT,
    PandaGpuManagerInitUnitTypeALL,
}

impl PandaGpuManager {
    pub fn new(device_id: usize) -> Result<Self, PandaGpuError> {
        let devices_num = get_device_number()?;
        if devices_num == 0 {
            return Err(PandaGpuError::GetDeviceCountError);
        }

        // Supports one GPU by default
        let mem_pool = Self::init_hardware(device_id).unwrap();

        Ok(Self {
            device_id,
            mem_pool,
            default_stream: PandaStream::new().unwrap(),
            h2d_stream: PandaStream::new().unwrap(),
            d2h_stream: PandaStream::new().unwrap(),
            exec_stream: PandaStream::new().unwrap(),
            d_bases: Vec::new(),
            d_scalars: Vec::new(),
            scalars_len: Vec::new(),
            msm_result_coordinate_type: PandaMSMResultCoordinateType::Jacobian,
        })
    }

    pub fn init_all(
        device_id: usize,
        init_unit_type: PandaGpuManagerInitUnitType,
        bases: Option<&[&[u8]]>,
        omega: Option<&[u8]>,
    ) -> Result<Self, PandaGpuError> {
        let devices_num = get_device_number()?;
        if devices_num == 0 {
            return Err(PandaGpuError::GetDeviceCountError);
        }

        let mem_pool = Self::init_hardware(device_id).unwrap();

        let mut d_bases_ptrs = vec![];
        match init_unit_type {
            PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeNone => {
                return Err(PandaGpuError::MSMBasesAddrError);
            }
            PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeMSM => {
                if let Some(datas) = bases {
                    d_bases_ptrs = Self::init_msm(datas).unwrap();
                } else {
                    return Err(PandaGpuError::MSMBasesAddrError);
                }
            }
            PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeNTT => {
                if let Some(data) = omega {
                    let _ = Self::init_ntt(data);
                } else {
                    return Err(PandaGpuError::NTTOmegaAddrError);
                }
            }
            PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeALL => {
                if let Some(datas) = bases {
                    d_bases_ptrs = Self::init_msm(datas).unwrap();
                } else {
                    return Err(PandaGpuError::MSMBasesAddrError);
                }
                if let Some(data) = omega {
                    let _ = Self::init_ntt(data);
                } else {
                    return Err(PandaGpuError::NTTOmegaAddrError);
                }
            }
        }

        Ok(Self {
            device_id,
            mem_pool,
            default_stream: PandaStream::new()?,
            h2d_stream: PandaStream::new()?,
            d2h_stream: PandaStream::new()?,
            exec_stream: PandaStream::new()?,
            d_bases: d_bases_ptrs,
            d_scalars: Vec::new(),
            scalars_len: Vec::new(),
            msm_result_coordinate_type: PandaMSMResultCoordinateType::Jacobian,
        })
    }

    pub fn init_hardware(device_id: usize) -> Result<PandaMemPool, PandaGpuError> {
        let _ = set_device(device_id);
        let mem_pool = PandaMemPool::new(device_id)?;

        Ok(mem_pool)
    }

    pub fn init_msm(bases: &[&[u8]]) -> Result<Vec<*mut c_void>, PandaGpuError> {
        let start_init_for_msm_multi_bases = start_timer!(|| "[panda][PandaGpuManager] init msm");

        let d_bases_ptrs: Result<Vec<*mut c_void>, PandaGpuError> = bases
            .iter()
            .map(|base| {
                let len = base.len() as u64;
                let mut d_bases_ptr = std::ptr::null_mut();

                let time1 =
                    start_timer!(|| "[panda][PandaGpuManager] malloc memcpy for bases data");
                let result = unsafe { panda_malloc(&mut d_bases_ptr, len) };
                if result != 0 {
                    return Err(PandaGpuError::CreateContextError);
                }
                if unsafe { panda_memcpy(d_bases_ptr, base.as_ptr() as *const c_void, len) } != 0 {
                    return Err(PandaGpuError::CreateContextError);
                }

                end_timer!(time1);
                Ok(d_bases_ptr)
            })
            .collect();

        if unsafe { panda_msm_setup_bn254() } != 0 {
            return Err(PandaGpuError::CreateContextError);
        }
        end_timer!(start_init_for_msm_multi_bases);

        Ok(d_bases_ptrs?)
    }

    pub fn init_msm_cached_bases(bases: &[u8]) -> Result<*mut c_void, PandaGpuError> {
        let len = bases.len() as u64;
        let mut d_bases_ptr = std::ptr::null_mut();

        let time1 = start_timer!(|| "[panda][PandaGpuManager] init_msm_cached_base");
        let result = unsafe { panda_malloc(&mut d_bases_ptr, len) };
        if result != 0 {
            return Err(PandaGpuError::CreateContextError);
        }
        if unsafe { panda_memcpy(d_bases_ptr, bases.as_ptr() as *const c_void, len) } != 0 {
            return Err(PandaGpuError::CreateContextError);
        }

        end_timer!(time1);
        Ok(d_bases_ptr)
    }

    pub fn init_msm_cached_scalars(scalars: &[u8]) -> Result<*mut c_void, PandaGpuError> {
        let len = scalars.len() as u64;
        let mut d_scalars_ptr = std::ptr::null_mut();

        let time1 = start_timer!(|| "[panda][PandaGpuManager] init_msm_cached_scalars");
        let result = unsafe { panda_malloc(&mut d_scalars_ptr, len) };
        if result != 0 {
            return Err(PandaGpuError::CreateContextError);
        }
        if unsafe { panda_memcpy(d_scalars_ptr, scalars.as_ptr() as *const c_void, len) } != 0 {
            return Err(PandaGpuError::CreateContextError);
        }

        end_timer!(time1);
        Ok(d_scalars_ptr)
    }

    pub fn init_msm_cached(
        scalars: &[u8],
        bases: &[u8],
    ) -> Result<(*mut c_void, *mut c_void), PandaGpuError> {
        let d_scalars_ptr = Self::init_msm_cached_scalars(scalars).unwrap();

        let d_bases_ptr = Self::init_msm_cached_bases(bases).unwrap();

        Ok((d_scalars_ptr, d_bases_ptr))
    }

    pub fn init_ntt(omega: &[u8]) -> Result<(), PandaGpuError> {
        let start_init_for_msm_multi_bases = start_timer!(|| "init_for_ntt");

        let omega_ptr = omega.as_ptr() as *mut c_void;

        if unsafe { panda_ntt_setup_bn254(omega_ptr) } != 0 {
            return Err(PandaGpuError::CreateContextError);
        }
        end_timer!(start_init_for_msm_multi_bases);

        Ok(())
    }

    pub fn set_config(&mut self, msm_result_coordinate_type: PandaMSMResultCoordinateType) {
        self.msm_result_coordinate_type = msm_result_coordinate_type;
    }

    pub fn get_mem_pool(&self) -> PandaMemPool {
        self.mem_pool
    }

    pub fn get_stream(&self) -> PandaStream {
        self.default_stream
    }
    pub fn get_h2d_stream(&self) -> PandaStream {
        self.h2d_stream
    }

    pub fn get_d2h_stream(&self) -> PandaStream {
        self.d2h_stream
    }

    pub fn get_exec_stream(&self) -> PandaStream {
        self.exec_stream
    }

    pub fn get_msm_result_coordinate_type(&self) -> PandaMSMResultCoordinateType {
        self.msm_result_coordinate_type
    }

    pub fn get_params_bases_ptr_mut(&self, index: usize) -> *mut c_void {
        match self.d_bases.get(index) {
            Some(ptr) => *ptr as *mut c_void,
            None => std::ptr::null_mut(),
        }
    }

    pub fn get_params_scalars_ptr_mut(&self, index: usize) -> *mut c_void {
        match self.d_scalars.get(index) {
            Some(ptr) => *ptr as *mut c_void,
            None => std::ptr::null_mut(),
        }
    }

    pub fn get_params_scalars_len(&self, index: usize) -> usize {
        match self.scalars_len.get(index) {
            Some(value) => *value,
            None => 0,
        }
    }

    pub fn wait_h2d(&self) -> Result<(), PandaGpuError> {
        let h2d_finished = PandaEvent::new()?;
        h2d_finished.record(self.get_h2d_stream())?;
        self.get_exec_stream().wait(h2d_finished)?;

        Ok(())
    }

    pub fn wait_exec(&self) -> Result<(), PandaGpuError> {
        let exec_finished = PandaEvent::new()?;
        exec_finished.record(self.get_exec_stream())?;
        self.get_d2h_stream().wait(exec_finished)?;
        Ok(())
    }

    pub fn destroy(&self) -> Result<(), PandaGpuError> {
        unsafe {
            if panda_mem_pool_destroy(self.get_mem_pool()) != 0 {
                return Err(PandaGpuError::DestroyContextErr);
            }
        }

        Ok(())
    }

    pub fn sync(&self) -> Result<(), PandaGpuError> {
        self.get_h2d_stream().sync()?;
        self.get_exec_stream().sync()?;
        self.get_d2h_stream().sync()?;

        Ok(())
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn deinit(&mut self) {
        if unsafe { panda_mem_pool_destroy(self.get_mem_pool()) } != 0 {
            panic!("couldn't destroy mempool");
        }

        for &d_bases in &self.d_bases {
            if unsafe { panda_free(d_bases as *mut c_void) } != 0 {
                panic!("couldn't free bases");
            }
            println!("d bases freed");
            if unsafe { panda_msm_tear_down() } != 0 {
                panic!("couldn't tear down msm");
            }
            println!("panda msm tear down");
        }
    }
}

pub fn get_device_number() -> Result<i32, PandaGpuError> {
    let mut count = 0;
    let success = unsafe { panda_get_device_number(std::ptr::addr_of_mut!(count)) } == 0;
    if success {
        Ok(count)
    } else {
        Err(PandaGpuError::GetDeviceCountError)
    }
}

pub fn device_info(device_id: i32) -> Result<PandaDeviceInfo, PandaGpuError> {
    let mut free = 0;
    let mut total = 0;
    let success = unsafe {
        let result = panda_set_device(device_id);
        assert_eq!(result, 0);
        panda_mem_get_info(std::ptr::addr_of_mut!(free), std::ptr::addr_of_mut!(total))
    } == 0;
    if success {
        Ok(PandaDeviceInfo { free, total })
    } else {
        Err(PandaGpuError::DeviceGetDeviceMemoryInfoError)
    }
}

pub fn set_device(device_id: usize) -> Result<(), PandaGpuError> {
    let success = unsafe { panda_set_device(device_id as i32) } == 0;
    if success {
        Ok(())
    } else {
        Err(PandaGpuError::SetDeviceError)
    }
}

impl fmt::Pointer for PandaGpuManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", self)
    }
}
