use std::marker::PhantomData;

use ark_std::{end_timer, start_timer};

use halo2_middleware::halo2curves::CurveAffine;

use panda::GLOBAL_DEVICE_MANAGER;
use panda::zal::traits::MsmAccel;

// GPU-> Panda Backend
// ---------------------------------------------------
#[derive(Default)]
pub struct PandaEngine;

pub struct PandaMsmCoeffsDesc<'b, C: CurveAffine> {
    raw: usize,
    _marker: PhantomData<&'b C>,
}
pub struct PandaMsmBaseDesc<'b, C: CurveAffine> {
    raw: usize,
    _marker: PhantomData<&'b C>,
}

impl PandaEngine {
    pub fn new() -> Self {
        Self {}
    }
}

impl<C: CurveAffine> MsmAccel<C> for PandaEngine {
    fn msm(&self, coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
        let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
        let device_manager_handle = binding.get_handle_mut();

        let t1 = start_timer!(|| format!("execute_msm"));
        let mut result_datas = device_manager_handle.execute_msm::<C>(coeffs, bases).unwrap();
        end_timer!(t1);

        let result_datas_ptr = result_datas.as_mut_ptr();
        let mut curve_value: C::Curve = Default::default();
        let size = std::mem::size_of::<u8>() * result_datas.len();
        unsafe {
            std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
        }

        let result = curve_value.clone();

        result
    }

    // Caching API
    // -------------------------------------------------
    type CoeffsDescriptor<'c> = GPUMsmCoeffsDesc<'c, C>;
    type BaseDescriptor<'b> = GPUMsmBaseDesc<'b, C>;

    fn get_coeffs_descriptor<'c>(&self, coeffs: &'c [C::Scalar]) -> Self::CoeffsDescriptor<'c> {
        let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
        let device_manager_handle = binding.get_handle_mut();

        let t1 = start_timer!(|| format!("init_msm_with_cached_scalars"));
        let scalars_id = device_manager_handle.init_msm_with_cached_scalars::<C>(coeffs).unwrap();
        end_timer!(t1);

        Self::CoeffsDescriptor { raw: scalars_id, _marker: PhantomData }
    }

    fn get_base_descriptor<'b>(&self, base: &'b [C]) -> Self::BaseDescriptor<'b> {
        let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
        let device_manager_handle = binding.get_handle_mut();

        let t1 = start_timer!(|| format!("init_msm_with_cached_bases"));
        let bases_id = device_manager_handle.init_msm_with_cached_bases::<C>(base).unwrap();
        end_timer!(t1);

        Self::BaseDescriptor { raw: bases_id, _marker: PhantomData }
    }

    fn msm_with_cached_scalars(
        &self,
        coeffs: &Self::CoeffsDescriptor<'_>,
        base: &[C],
    ) -> C::Curve {

        let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
        let device_manager_handle = binding.get_handle_mut();

        let t1 = start_timer!(|| format!("execute_msm_with_cached_scalars"));
        let mut result_datas = device_manager_handle.
            execute_msm_with_cached_scalars::<C>(coeffs.raw, base).unwrap();
        end_timer!(t1);

        let result_datas_ptr = result_datas.as_mut_ptr();
        let mut curve_value: C::Curve = Default::default();
        let size = std::mem::size_of::<u8>() * result_datas.len();
        unsafe {
            std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
        }

        let result = curve_value.clone();

        result
    }

    fn msm_with_cached_base(
        &self,
        coeffs: &[C::Scalar],
        base: &Self::BaseDescriptor<'_>,
    ) -> C::Curve {

        let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
        let device_manager_handle = binding.get_handle_mut();

        let t1 = start_timer!(|| format!("execute_msm_with_cached_bases"));
        let mut result_datas = device_manager_handle.
            execute_msm_with_cached_bases::<C>(coeffs, base.raw).unwrap();
        end_timer!(t1);

        let result_datas_ptr = result_datas.as_mut_ptr();
        let mut curve_value: C::Curve = Default::default();
        let size = std::mem::size_of::<u8>() * result_datas.len();
        unsafe {
            std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
        }

        let result = curve_value.clone();

        result
    }

    fn msm_with_cached_inputs(
        &self,
        coeffs: &Self::CoeffsDescriptor<'_>,
        base: &Self::BaseDescriptor<'_>,
    ) -> C::Curve {
        let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
        let device_manager_handle = binding.get_handle_mut();

        let t1 = start_timer!(|| format!("execute_msm_with_cached_input"));
        let mut result_datas = device_manager_handle.
            execute_msm_with_cached_input::<C>(coeffs.raw, base.raw).unwrap();
        end_timer!(t1);

        let result_datas_ptr = result_datas.as_mut_ptr();
        let mut curve_value: C::Curve = Default::default();
        let size = std::mem::size_of::<u8>() * result_datas.len();
        unsafe {
            std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
        }

        let result = curve_value.clone();

        result
    }
}