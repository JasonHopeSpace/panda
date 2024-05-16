use super::*;
use std::ffi::c_void;
use std::ptr::addr_of_mut;

pub fn log_2(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

pub fn malloc_from_pool_async(
    ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
    pool: PandaMemPool,
    stream: PandaStream,
) -> Result<(), PandaGpuError> {
    let result = unsafe { panda_malloc_from_pool_async(ptr, size as SizeT, pool, stream) };

    if result != 0 {
        return Err(PandaGpuError::AsyncPoolMallocErr);
    }

    Ok(())
}

pub fn memcpy_async(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    size: usize,
    stream: PandaStream,
) -> Result<(), PandaGpuError> {
    if unsafe { panda_memcpy_async(dst, src, size as u64, stream) } != 0 {
        return Err(PandaGpuError::AsyncMemcopyErr);
    }
    Ok(())
}

pub fn free_async(
    ptr: *mut ::std::os::raw::c_void,
    stream: PandaStream,
) -> Result<(), PandaGpuError> {
    if unsafe { panda_free_async(ptr, stream) } != 0 {
        return Err(PandaGpuError::AsyncMemcopyErr);
    }
    Ok(())
}

pub fn memory_alloc_and_copy(
    gm: &PandaGpuManager,
    h_values: &[u8],
    stream: PandaStream,
) -> Result<*mut c_void, PandaGpuError> {
    let len = h_values.len();
    let mut d_values = std::ptr::null_mut();
    malloc_from_pool_async(addr_of_mut!(d_values), len, gm.get_mem_pool(), stream).unwrap();
    memcpy_async(d_values, h_values.as_ptr() as *const c_void, len, stream).unwrap();

    Ok(d_values)
}

pub fn memory_copy_and_free(
    h_values: &mut [u8],
    d_values: *mut c_void,
    stream: PandaStream,
) -> Result<(), PandaGpuError> {
    let len = h_values.len();
    memcpy_async(h_values.as_ptr() as *mut c_void, d_values, len, stream)?;
    free_async(d_values, stream)?;
    Ok(())
}
