// finite field definitions

#pragma once

#include "field/field_storage.cuh"
#include "config.cuh"

namespace CURVE_BLS12_377 
{
  #define NLIMBS(bits)   (bits/LIMB_T_BITS)


  #if PANDA_ASM_32
    // fp
  struct fp_configuration
  {
    // field structure size = 6 * 64 bit
    static constexpr unsigned bits_count = 377;
    static constexpr unsigned limbs_bits = 32;
    static constexpr unsigned limbs_count = 12;
    static constexpr unsigned bytes_count = 48;

    static constexpr field_storage<limbs_count> MODULAR = 
    {
        0x00000001,
        0x8508c000,
        0x30000000,
        0x170b5d44,
        0xba094800,
        0x1ef3622f,
        0x00f5138f,
        0x1a22d9f3,
        0x6ca1493b,
        0xc63b05c0,
        0x17c510ea,
        0x01ae3a46
    };

  static constexpr uint32_t MODULAR_INV = 0xffffffff;

    static constexpr field_storage<limbs_count> R1 = 
    {
        0xffffff68,
        0x02cdffff,
        0x7fffffb1,
        0x51409f83,
        0x8a7d3ff2,
        0x9f7db3a9,
        0x6e7c6305,
        0x7b4e97b7,
        0x803c84e8,
        0x4cf495bf,
        0xe2fdf49a,
        0x008d6661
    };
    static constexpr field_storage<limbs_count> ZERO = {0x0};
    static constexpr field_storage<limbs_count> ONE { R1 };

    static constexpr field_storage<limbs_count> R2 = 
    {
        0x9400cd22,
        0xb786686c,
        0xb00431b1,
        0x0329fcaa,
        0x62d6b46d,
        0x22a5f111,
        0x827dc3ac,
        0xbfdf7d03,
        0x41790bf9,
        0x837e92f0,
        0x1e914b88,
        0x006dfccb
    };

    static constexpr field_storage<limbs_count> UNITY_ONE { 0x1 };
  };
  #else
  // fp
  struct fp_configuration
  {
    // field structure size = 6 * 64 bit
    static constexpr unsigned bits_count = 377;
    static constexpr unsigned limbs_bits = 64;
    static constexpr unsigned limbs_count = 6;
    static constexpr unsigned bytes_count = 48;
  
    static constexpr field_storage<limbs_count> MODULAR = 
    {
        0x8508c00000000001,
        0x170b5d4430000000,
        0x1ef3622fba094800,
        0x1a22d9f300f5138f,
        0xc63b05c06ca1493b,
        0x01ae3a4617c510ea
    };

    // modulus inv
    static constexpr uint64_t MODULAR_INV = 0x8508bfffffffffff;
  
    static constexpr field_storage<limbs_count> R1 = 
    {
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a
    };

    static constexpr field_storage<limbs_count> ZERO = {0x0};
    static constexpr field_storage<limbs_count> ONE { R1 };

    static constexpr field_storage<limbs_count> R2 = 
    {
        0xb786686c9400cd22,
        0x0329fcaab00431b1,
        0x22a5f11162d6b46d,
        0xbfdf7d03827dc3ac,
        0x837e92f041790bf9,
        0x006dfccb1e914b88,
    };

    static constexpr field_storage<limbs_count> UNITY_ONE { 0x1 };
  };
  #endif


// fr
#if PANDA_ASM_32
  struct fr_configuration
  {
    // field structure size = 4 * 64 bit
    static constexpr unsigned bits_count = 253;
    static constexpr unsigned limbs_bits = 32;
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned bytes_count = 32;

    static constexpr field_storage<limbs_count> MODULAR = 
    {
        0x00000001,
        0x0A118000,
        0xD0000001,
        0x59AA76FE,
        0x5C37B001,
        0x60B44D1E,
        0x9A2CA556,
        0x12AB655E
    };

    // modulus inv
    static constexpr uint32_t MODULAR_INV = 0xFFFFFFFF;

    static constexpr field_storage<limbs_count> R1 = 
    {
        0xfffffff3,
        0x7d1c7fff,
        0x6ffffff2,
        0x7257f50f,
        0x512c0fee,
        0x16d81575,
        0x2bbb9a9d,
        0x0d4bda32
    };

    static constexpr field_storage<limbs_count> ZERO = {0x0};
    static constexpr field_storage<limbs_count> ONE { R1 };

    static constexpr field_storage<limbs_count> R2 = 
    {
        0xB861857B,
        0x25D577BA,
        0x8860591F,
        0xCC2C27B5,
        0xE5DC8593,
        0xA7CC008F,
        0xEFF1C939,
        0x011FDAE7

    };
    static constexpr field_storage<limbs_count> UNITY_ONE { 0x1 };
  };


#else
  struct fr_configuration
  {
    // field structure size = 4 * 64 bit
    static constexpr unsigned bits_count = 253;
    static constexpr unsigned limbs_bits = 64;
    static constexpr unsigned limbs_count = 4;
    static constexpr unsigned bytes_count = 32;

    static constexpr field_storage<limbs_count> MODULAR = 
    {
        0x0A11800000000001,
        0x59AA76FED0000001,
        0x60B44D1E5C37B001,
        0x12AB655E9A2CA556
    };

    // modulus inv
    static constexpr uint64_t MODULAR_INV = 0x0A117FFFFFFFFFFF;

    static constexpr field_storage<limbs_count> R1 = 
    {
        0x7d1c7ffffffffff3,
        0x7257f50f6ffffff2,
        0x16d81575512c0fee,
        0x0d4bda322bbb9a9d
    };

    static constexpr field_storage<limbs_count> ZERO = {0x0};
    static constexpr field_storage<limbs_count> ONE { R1 };

    static constexpr field_storage<limbs_count> R2 = 
    {
        0x25D577BAB861857B,
        0xCC2C27B58860591F,
        0xA7CC008FE5DC8593,
        0x011FDAE7EFF1C939
    };
    static constexpr field_storage<limbs_count> UNITY_ONE { 0x1 };
  };
#endif
  static constexpr unsigned WEIERSTRASS_B = 3;
}
