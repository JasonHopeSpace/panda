#pragma once

#include <cstdint>
#include "math_64bit.cuh"
#include "field/asm/ptx.cuh"

namespace common
{
#if PANDA_ASM_32
  template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false> struct carry_chain
  {
    unsigned index;

    constexpr __host__ __device__ __forceinline__ carry_chain() : index(0) {}

    __device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y)
    {

      index++;

      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::add(x, y);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::add_cc(x, y);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::addc_cc(x, y);
      }
      else
      {
        return ptx::addc(x, y);
      }
    }

    __device__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y)
    {
  
      index++;
  
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::sub(x, y);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::sub_cc(x, y);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::subc_cc(x, y);
      }
      else
      {
        return ptx::subc(x, y);
      }
    }

    __device__ __forceinline__ uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z)
    {

      index++;

      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::mad_lo(x, y, z);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::mad_lo_cc(x, y, z);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::madc_lo_cc(x, y, z);
      }
      else
      {
        return ptx::madc_lo(x, y, z);
      }
    }

    __device__ __forceinline__ uint32_t mad_hi(const uint32_t x, const uint32_t y, const uint32_t z)
    {

      index++;

      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::mad_hi(x, y, z);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::mad_hi_cc(x, y, z);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::madc_hi_cc(x, y, z);
      }
      else
      {
        return ptx::madc_hi(x, y, z);
      }
    }
  };
#else
// #elif PANDA_ASM_64
  template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false> struct carry_chain
  {
    unsigned index;

    constexpr __host__ __device__ __forceinline__ carry_chain() : index(0) {}

    __device__ __forceinline__ uint64_t add(const uint64_t x, const uint64_t y)
    {

      index++;

      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::u64::add(x, y);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::u64::add_cc(x, y);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::u64::addc_cc(x, y);
      }
      else
      {
        return ptx::u64::addc(x, y);
      }
    }

    __device__ __forceinline__ uint64_t sub(const uint64_t x, const uint64_t y)
    {
  
      index++;
  
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::u64::sub(x, y);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::u64::sub_cc(x, y);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::u64::subc_cc(x, y);
      }
      else
      {
        return ptx::u64::subc(x, y);
      }
    }

    __device__ __forceinline__ uint64_t mad_lo(const uint64_t x, const uint64_t y, const uint64_t z)
    {

      index++;

      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::u64::mad_lo(x, y, z);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::u64::mad_lo_cc(x, y, z);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::u64::madc_lo_cc(x, y, z);
      }
      else
      {
        return ptx::u64::madc_lo(x, y, z);
      }
    }

    __device__ __forceinline__ uint64_t mad_hi(const uint64_t x, const uint64_t y, const uint64_t z)
    {

      index++;

      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      {
        return ptx::u64::mad_hi(x, y, z);
      }
      else if (index == 1 && !CARRY_IN)
      {
        return ptx::u64::mad_hi_cc(x, y, z);
      }
      else if (index < OPS_COUNT || CARRY_OUT)
      {
        return ptx::u64::madc_hi_cc(x, y, z);
      }
      else
      {
        return ptx::u64::madc_hi(x, y, z);
      }
    }
  };
#endif
}