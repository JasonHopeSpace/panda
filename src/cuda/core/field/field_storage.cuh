#pragma once

#include <cstdint>
#include "../common/common.cuh"

using namespace std;
using namespace common;

#define TO_LIMB_T(x)  x

//template <unsigned LIMBS_COUNT> struct __align__(TO_LIMB_T(LIMBS_COUNT)) field_storage
template <unsigned LIMBS_COUNT> struct __align__(8) field_storage
{
  static constexpr unsigned LC = LIMBS_COUNT;
  limb_t limbs[LIMBS_COUNT];
};

//template <unsigned LIMBS_COUNT> struct __align__(TO_LIMB_T(LIMBS_COUNT)) field_storage_wide
template <unsigned LIMBS_COUNT> struct __align__(8) field_storage_wide
{
  static_assert(LIMBS_COUNT ^ 1);
  static constexpr unsigned LC = LIMBS_COUNT;
  static constexpr unsigned LC2 = LIMBS_COUNT * 2;
  limb_t limbs[LC2];

  void __device__ __forceinline__ set_lo(const field_storage<LIMBS_COUNT> &in)
  {
    #pragma unroll
    for (unsigned i = 0; i < LC; i++)
    {
      limbs[i] = in.limbs[i];
    }
  }

  void __device__ __forceinline__ set_hi(const field_storage<LIMBS_COUNT> &in)
  {
    #pragma unroll
    for (unsigned i = 0; i < LC; i++)
    {
      limbs[i + LC].x = in.limbs[i];
    }
  }

  field_storage<LC> __device__ __forceinline__ get_lo()
  {
    field_storage<LC> out{};

    #pragma unroll
    for (unsigned i = 0; i < LC; i++)
    {
      out.limbs[i] = limbs[i];
    }

    return out;
  }

  field_storage<LC> __device__ __forceinline__ get_hi()
  {
    field_storage<LC> out{};

    #pragma unroll
    for (unsigned i = 0; i < LC; i++)
    {
      out.limbs[i] = limbs[i + LC];
    }

    return out;
  }
};
