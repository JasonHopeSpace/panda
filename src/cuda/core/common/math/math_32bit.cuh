
#pragma once

#include <stdio.h>
#include <cstdint>
#include <math.h>
#include <inttypes.h>

#include "math.cuh"

namespace MATH_32BIT
{
    #define UINT32_BORROW ((uint64_t)pow(2,32))

    inline uint32_t add_with_carry_in(uint32_t a, uint32_t b, uint32_t *carry)
    {
        uint64_t tmp = (uint64_t)a + (uint64_t)b + (uint64_t)*carry;

        *carry = (uint32_t)(tmp >> 32);

        return (uint32_t)tmp;
    }

    inline uint32_t add_carry_with_carry(uint32_t a, uint32_t *carry)
    {
        uint64_t tmp = (uint64_t)a  + (uint64_t)*carry;

        *carry = (uint32_t)(tmp >> 32);

        return (uint32_t)tmp;
    }

    inline uint32_t sub(uint32_t a, uint32_t b)
    {
        uint64_t c = (uint64_t)a - (uint64_t)b;

        return (uint32_t)c;
    }

    inline uint32_t sub_with_borrow(uint32_t a, uint32_t b, uint32_t *borrow)
    {
        uint32_t tmp = 0x0;
        uint32_t c = 0x0;
        uint32_t carry_in = *borrow;
        uint32_t carry_out = 0;

        /* case: 0 - 1  */
        if ((0x0 == a) && (1 == carry_in))
        {
            tmp = (uint32_t)(UINT32_BORROW - 1);
            carry_out = 1;
        }
        else    /* a >= tmp */
        {
            tmp = a - carry_in;
        }

        if ( tmp >= b)
        {
            c = sub(tmp, b);
        }
        else
        {
            c = (uint32_t)((uint64_t)tmp + UINT64_BORROW - (uint64_t)b);
            carry_out = 1;
        }

        *borrow = carry_out;
        return c;
    }

    inline uint32_t mul_with_carry(uint32_t a, uint32_t b, uint32_t *carry)
    {
        uint64_t tmp = (uint64_t)a * (uint64_t)b;

        *carry = (uint32_t)(tmp >> 32);

        return (uint32_t)tmp;
    }

    inline uint32_t mac_with_carry(uint32_t a, uint32_t b, uint32_t c, uint32_t *carry)
    {
        uint64_t tmp = (uint64_t)a * (uint64_t)b + (uint64_t)c;

        *carry = (uint32_t)(tmp >> 32);

        return (uint32_t)tmp;
    }

    inline uint32_t mac_with_carry_in(uint32_t a, uint32_t b, uint32_t c, uint32_t *carry)
    {
        uint64_t tmp = (uint64_t)a * (uint64_t)b + (uint64_t)c + (uint64_t)*carry;

        *carry = (uint32_t)(tmp >> 32);

        return (uint32_t)tmp;
    }
}
