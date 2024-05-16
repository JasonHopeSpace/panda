//pub mod arithmetic;

pub mod common;
pub mod unit;
pub mod wrapper;

pub use crate::gpu_ffi::binding::*;
pub use crate::gpu_ffi::common::*;
pub use crate::gpu_ffi::*;
pub use wrapper::*;

pub const BN254_SCALAR_WIDTH_BITS: usize = 254;
pub const BN254_POINT_WIDTH_BITS: usize = 254;
pub const FIELD_ELEMENT_LEN: usize = 32;
pub const TRUE: u32 = 1;
pub const FALSE: u32 = 0;
