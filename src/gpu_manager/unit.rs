use ark_std::{end_timer, start_timer};
use std::convert::TryInto;
use std::ffi::c_void;
use std::ptr::addr_of_mut;

use super::FIELD_ELEMENT_LEN;
use super::*;
use common::*;

pub fn panda_msm_bn254_gpu(
    gm: &PandaGpuManager,
    scalars: &[u8],
    bases: &[u8],
) -> Result<Vec<u8>, PandaGpuError> {
    let len = scalars.len();

    let time1 = start_timer!(|| "[panda msm] alloc_and_copy scalars");
    let d_scalars = memory_alloc_and_copy(gm, scalars, gm.get_h2d_stream()).unwrap();
    let h2d_finished = PandaEvent::new()?;
    h2d_finished.record(gm.get_h2d_stream())?;
    gm.get_exec_stream().wait(h2d_finished)?;
    end_timer!(time1);

    let time1 = start_timer!(|| "[panda msm] alloc_and_copy bases");
    let d_bases = memory_alloc_and_copy(gm, bases, gm.get_h2d_stream()).unwrap();
    let h2d_finished = PandaEvent::new()?;
    h2d_finished.record(gm.get_h2d_stream())?;
    gm.get_exec_stream().wait(h2d_finished)?;
    end_timer!(time1);

    let log_scalars_count = log_2((len / FIELD_ELEMENT_LEN) as usize);
    let result_buf_len = FIELD_ELEMENT_LEN * 3;
    let mut d_result = std::ptr::null_mut();

    malloc_from_pool_async(
        addr_of_mut!(d_result),
        result_buf_len,
        gm.get_mem_pool(),
        gm.get_h2d_stream(),
    )?;

    let cfg = MSMConfiguration {
        mem_pool: gm.get_mem_pool(),
        stream: gm.get_exec_stream(),
        bases: d_bases,
        scalars: d_scalars,
        results: d_result,
        log_scalars_count,
        msm_result_coordinate_type: gm.get_msm_result_coordinate_type(),
    };

    let time1 = start_timer!(|| "[panda msm] panda_msm_execute_bn254");

    unsafe {
        if panda_msm_execute_bn254(cfg) != 0 {
            return Err(PandaGpuError::SchedulingErr);
        };
    }

    let exec_finished = PandaEvent::new()?;
    exec_finished.record(gm.get_exec_stream())?;
    let _ = exec_finished.sync();
    end_timer!(time1);

    let time1 = start_timer!(|| "[panda msm] panda_malloc_host panda_memcpy");

    let mut result_ptr = std::ptr::null_mut();

    let ret = unsafe {
        panda_malloc_host(
            std::ptr::addr_of_mut!(result_ptr),
            result_buf_len.try_into().unwrap(),
        )
    };
    if ret != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_memcpy(result_ptr, d_result, result_buf_len.try_into().unwrap()) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    let result_slice =
        unsafe { std::slice::from_raw_parts_mut(result_ptr as *mut u8, result_buf_len) };

    if unsafe { panda_free(d_scalars) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_free(d_bases) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_free(d_result) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    end_timer!(time1);

    Ok(result_slice.to_vec())
}

pub fn panda_msm_bn254_gpu_with_cached_bases(
    gm: &PandaGpuManager,
    scalars: &[u8],
    bases_index: usize,
) -> Result<Vec<u8>, PandaGpuError> {
    let len = scalars.len();
    let time1 = start_timer!(|| "[panda msm] alloc_and_copy scalars");
    let d_scalars = memory_alloc_and_copy(gm, scalars, gm.get_h2d_stream())?;
    let h2d_finished = PandaEvent::new()?;
    h2d_finished.record(gm.get_h2d_stream())?;
    gm.get_exec_stream().wait(h2d_finished)?;
    end_timer!(time1);

    let log_scalars_count = log_2((len / FIELD_ELEMENT_LEN) as usize);
    let result_buf_len = FIELD_ELEMENT_LEN * 3;
    let mut d_result = std::ptr::null_mut();

    malloc_from_pool_async(
        addr_of_mut!(d_result),
        result_buf_len,
        gm.get_mem_pool(),
        gm.get_h2d_stream(),
    )?;

    let mut cfg = MSMConfiguration {
        mem_pool: gm.get_mem_pool(),
        stream: gm.get_exec_stream(),
        bases: std::ptr::null_mut(),
        scalars: d_scalars,
        results: d_result,
        log_scalars_count,
        msm_result_coordinate_type: gm.get_msm_result_coordinate_type(),
    };

    if gm.get_params_bases_ptr_mut(bases_index) == std::ptr::null_mut() {
        return Err(PandaGpuError::BasesIndexErr);
    } else {
        cfg.bases = gm.get_params_bases_ptr_mut(bases_index);
    }

    let time1 = start_timer!(|| "[panda msm] panda_msm_execute_bn254");

    unsafe {
        if panda_msm_execute_bn254(cfg) != 0 {
            return Err(PandaGpuError::SchedulingErr);
        };
    }

    let exec_finished = PandaEvent::new()?;
    exec_finished.record(gm.get_exec_stream())?;
    let _ = exec_finished.sync();
    end_timer!(time1);

    let time1 = start_timer!(|| "[panda msm] panda_malloc_host panda_memcpy");

    let mut result_ptr = std::ptr::null_mut();

    let ret = unsafe {
        panda_malloc_host(
            std::ptr::addr_of_mut!(result_ptr),
            result_buf_len.try_into().unwrap(),
        )
    };
    if ret != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_memcpy(result_ptr, d_result, result_buf_len.try_into().unwrap()) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    let result_slice =
        unsafe { std::slice::from_raw_parts_mut(result_ptr as *mut u8, result_buf_len) };

    if unsafe { panda_free(d_scalars) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_free(d_result) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    end_timer!(time1);

    Ok(result_slice.to_vec())
}

pub fn panda_msm_bn254_gpu_with_cached_scalars(
    gm: &PandaGpuManager,
    scalars_index: usize,
    bases: &[u8],
) -> Result<Vec<u8>, PandaGpuError> {
    let len = bases.len();
    let time1 = start_timer!(|| "[panda msm] alloc_and_copy bases");
    let d_bases = memory_alloc_and_copy(gm, bases, gm.get_h2d_stream())?;
    let h2d_finished = PandaEvent::new()?;
    h2d_finished.record(gm.get_h2d_stream())?;
    gm.get_exec_stream().wait(h2d_finished)?;
    end_timer!(time1);

    let log_scalars_count = log_2((len / FIELD_ELEMENT_LEN / 2) as usize);
    let result_buf_len = FIELD_ELEMENT_LEN * 3;
    let mut d_result = std::ptr::null_mut();

    malloc_from_pool_async(
        addr_of_mut!(d_result),
        result_buf_len,
        gm.get_mem_pool(),
        gm.get_h2d_stream(),
    )?;

    let mut cfg = MSMConfiguration {
        mem_pool: gm.get_mem_pool(),
        stream: gm.get_exec_stream(),
        bases: d_bases,
        scalars: std::ptr::null_mut(),
        results: d_result,
        log_scalars_count,
        msm_result_coordinate_type: gm.get_msm_result_coordinate_type(),
    };

    if gm.get_params_scalars_ptr_mut(scalars_index) == std::ptr::null_mut() {
        return Err(PandaGpuError::BasesIndexErr);
    } else {
        cfg.scalars = gm.get_params_scalars_ptr_mut(scalars_index);
    }

    let time1 = start_timer!(|| "[panda msm] panda_msm_execute_bn254");

    unsafe {
        if panda_msm_execute_bn254(cfg) != 0 {
            return Err(PandaGpuError::SchedulingErr);
        };
    }

    let exec_finished = PandaEvent::new()?;
    exec_finished.record(gm.get_exec_stream())?;
    let _ = exec_finished.sync();
    end_timer!(time1);

    let time1 = start_timer!(|| "[panda msm] panda_malloc_host panda_memcpy");

    let mut result_ptr = std::ptr::null_mut();

    let ret = unsafe {
        panda_malloc_host(
            std::ptr::addr_of_mut!(result_ptr),
            result_buf_len.try_into().unwrap(),
        )
    };
    if ret != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_memcpy(result_ptr, d_result, result_buf_len.try_into().unwrap()) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    let result_slice =
        unsafe { std::slice::from_raw_parts_mut(result_ptr as *mut u8, result_buf_len) };

    if unsafe { panda_free(d_bases) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_free(d_result) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    end_timer!(time1);

    Ok(result_slice.to_vec())
}

pub fn panda_msm_bn254_gpu_with_cached_input(
    gm: &PandaGpuManager,
    scalars_index: usize,
    bases_index: usize,
) -> Result<Vec<u8>, PandaGpuError> {
    let len = gm.get_params_scalars_len(scalars_index);
    if len == 0 {
        return Err(PandaGpuError::BasesIndexErr);
    }

    let log_scalars_count = log_2((len / FIELD_ELEMENT_LEN) as usize);
    let result_buf_len = FIELD_ELEMENT_LEN * 3;
    let mut d_result = std::ptr::null_mut();

    malloc_from_pool_async(
        addr_of_mut!(d_result),
        result_buf_len,
        gm.get_mem_pool(),
        gm.get_h2d_stream(),
    )?;

    let mut cfg = MSMConfiguration {
        mem_pool: gm.get_mem_pool(),
        stream: gm.get_exec_stream(),
        bases: std::ptr::null_mut(),
        scalars: std::ptr::null_mut(),
        results: d_result,
        log_scalars_count,
        msm_result_coordinate_type: gm.get_msm_result_coordinate_type(),
    };

    if gm.get_params_scalars_ptr_mut(scalars_index) == std::ptr::null_mut() {
        return Err(PandaGpuError::BasesIndexErr);
    } else {
        cfg.scalars = gm.get_params_scalars_ptr_mut(scalars_index);
    }

    if gm.get_params_bases_ptr_mut(bases_index) == std::ptr::null_mut() {
        return Err(PandaGpuError::BasesIndexErr);
    } else {
        cfg.bases = gm.get_params_bases_ptr_mut(bases_index);
    }

    let time1 = start_timer!(|| "[panda msm] panda_msm_execute_bn254");

    unsafe {
        if panda_msm_execute_bn254(cfg) != 0 {
            return Err(PandaGpuError::SchedulingErr);
        };
    }

    let exec_finished = PandaEvent::new()?;
    exec_finished.record(gm.get_exec_stream())?;
    let _ = exec_finished.sync();
    end_timer!(time1);

    let time1 = start_timer!(|| "[panda msm] panda_malloc_host panda_memcpy");

    let mut result_ptr = std::ptr::null_mut();

    let ret = unsafe {
        panda_malloc_host(
            std::ptr::addr_of_mut!(result_ptr),
            result_buf_len.try_into().unwrap(),
        )
    };
    if ret != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_memcpy(result_ptr, d_result, result_buf_len.try_into().unwrap()) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    let result_slice =
        unsafe { std::slice::from_raw_parts_mut(result_ptr as *mut u8, result_buf_len) };

    if unsafe { panda_free(d_result) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    end_timer!(time1);

    Ok(result_slice.to_vec())
}

pub fn panda_msm_bn254_gpu_host(
    gm: &PandaGpuManager,
    scalars: &[u8],
    // bases_index: usize,
    bases: &[u8],
) -> Result<Vec<u8>, PandaGpuError> {
    let len = scalars.len();

    let log_scalars_count = log_2((len / FIELD_ELEMENT_LEN) as usize);
    let result_buf_len = FIELD_ELEMENT_LEN * 3;

    let mut result_ptr = std::ptr::null_mut();

    let ret = unsafe {
        panda_malloc_host(
            std::ptr::addr_of_mut!(result_ptr),
            result_buf_len.try_into().unwrap(),
        )
    };
    if ret != 0 {
        return Err(PandaGpuError::CreateContextError);
    }
    let cfg = MSMConfiguration {
        mem_pool: gm.get_mem_pool(),
        stream: gm.get_exec_stream(),
        bases: bases.as_ptr() as *mut c_void,
        scalars: scalars.as_ptr() as *mut c_void,
        results: result_ptr,
        log_scalars_count,
        msm_result_coordinate_type: gm.get_msm_result_coordinate_type(),
    };

    let time1 = start_timer!(|| "[panda msm] panda_msm_execute_bn254");

    unsafe {
        if panda_msm_execute_bn254_host(cfg) != 0 {
            return Err(PandaGpuError::SchedulingErr);
        };
    }

    let exec_finished = PandaEvent::new()?;
    exec_finished.record(gm.get_exec_stream())?;
    let _ = exec_finished.sync();
    end_timer!(time1);

    let time1 = start_timer!(|| "[panda msm] panda_malloc_host panda_memcpy");

    let result_slice =
        unsafe { std::slice::from_raw_parts_mut(result_ptr as *mut u8, result_buf_len) };

    end_timer!(time1);

    Ok(result_slice.to_vec())
}

pub fn panda_ntt_bn254_gpu(
    gm: &PandaGpuManager,
    scalars: &mut [u8],
    log_n: u32,
) -> Result<(), PandaGpuError> {
    assert_eq!(scalars.len(), (1 << log_n) * 32);

    let time1 = start_timer!(|| "[panda ntt] alloc_and_copy scalars");
    let d_src = memory_alloc_and_copy(gm, scalars, gm.get_h2d_stream())?;
    end_timer!(time1);

    let mut d_dst = std::ptr::null_mut();
    let scalars_buffer_len = scalars.len();

    malloc_from_pool_async(
        addr_of_mut!(d_dst),
        scalars_buffer_len,
        gm.get_mem_pool(),
        gm.get_h2d_stream(),
    )?;

    let mut flag = 0u32;
    let cfg = NTTConfiguration {
        mem_pool: gm.get_mem_pool(),
        stream: gm.get_exec_stream(),
        d_src,
        d_dst,
        log_n,
        flag: &mut flag as *mut std::os::raw::c_uint,
    };

    let time1 = start_timer!(|| "[panda ntt] panda_ntt_execute_bn254");
    unsafe {
        if panda_ntt_execute_bn254(cfg) != 0 {
            return Err(PandaGpuError::SchedulingErr);
        };
    }
    end_timer!(time1);

    let scalars_ptr: *mut c_void = scalars.as_mut_ptr() as *mut c_void;
    if flag == 0 {
        if unsafe { panda_memcpy(scalars_ptr, d_src, scalars_buffer_len.try_into().unwrap()) } != 0
        {
            return Err(PandaGpuError::CreateContextError);
        }
    } else if flag == 1 {
        if unsafe { panda_memcpy(scalars_ptr, d_dst, scalars_buffer_len.try_into().unwrap()) } != 0
        {
            return Err(PandaGpuError::CreateContextError);
        }
    }

    if unsafe { panda_free(d_src) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_free(d_dst) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    Ok(())
}

pub fn panda_ntt_bn254_gpu_v1(
    gm: &PandaGpuManager,
    scalars: &mut [u8],
    omega: &[u8],
    log_n: u32,
) -> Result<(), PandaGpuError> {
    let time1 = start_timer!(|| "[panda ntt] alloc_and_copy scalars");
    let d_src = memory_alloc_and_copy(gm, scalars, gm.get_h2d_stream())?;
    end_timer!(time1);

    let mut d_dst = std::ptr::null_mut();
    let scalars_buffer_len = scalars.len();

    malloc_from_pool_async(
        addr_of_mut!(d_dst),
        scalars_buffer_len,
        gm.get_mem_pool(),
        gm.get_h2d_stream(),
    )?;

    let h_omega = omega.as_ptr() as *const c_void;
    let mut flag = 0u32;
    let cfg = NttconfigurationV1 {
        mem_pool: gm.get_mem_pool(),
        stream: gm.get_exec_stream(),
        d_src,
        d_dst,
        omega: h_omega,
        log_n,
        flag: &mut flag as *mut std::os::raw::c_uint,
    };

    let time1 = start_timer!(|| "[panda ntt] panda_ntt_bn254_gpu_v1");
    unsafe {
        if panda_ntt_execute_bn254_v1(cfg) != 0 {
            return Err(PandaGpuError::SchedulingErr);
        };
    }
    end_timer!(time1);

    let scalars_ptr: *mut c_void = scalars.as_mut_ptr() as *mut c_void;
    if flag == 0 {
        if unsafe { panda_memcpy(scalars_ptr, d_src, scalars_buffer_len.try_into().unwrap()) } != 0
        {
            return Err(PandaGpuError::CreateContextError);
        }
    } else if flag == 1 {
        if unsafe { panda_memcpy(scalars_ptr, d_dst, scalars_buffer_len.try_into().unwrap()) } != 0
        {
            return Err(PandaGpuError::CreateContextError);
        }
    }

    if unsafe { panda_free(d_src) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    if unsafe { panda_free(d_dst) } != 0 {
        return Err(PandaGpuError::CreateContextError);
    }

    Ok(())
}
