

#pragma once

#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "field/field.cuh"
#include "field/field_storage.cuh"
#include "common/common.cuh"

// typedef unsigned long long limb_t;
using namespace common;

namespace ASM_CUDA_FP384 {

    #ifdef __CUDA_ARCH__
    
    typedef field_storage<6> FS;

    template <class storage>
    __device__ void mul_mont_384(storage &ret, const storage &a, const storage &b, const storage &m, limb_t m_inv);

    template <class storage>
    __device__ void sqr_mont_384(storage &ret, const storage &a, const storage &m, limb_t m_inv);

    template <class storage>
    __device__ void add_mod_384(storage &ret, const storage &a, const storage &b, const storage &m);

    template <class storage>
    __device__ void sub_mod_384(storage &ret, const storage &a, const storage &b, const storage &m);

    template <class storage>
    __device__ void sub_384_unsafe(storage &ret, const storage &a, const storage &b);

    template <class storage>
    __device__ void add_384_unsafe(storage &ret, const storage &a, const storage &b);

    template <class storage>
    __device__ void div_by_2_mod_384(storage &ret, const storage &a);

    template <class storage>
    __device__ void cneg_mod_384(storage &ret, const storage &a, bool flag, const storage &m) ;

    template <class storage>
    __device__ storage inverse(storage &u, storage &v, storage &b, storage &c, const storage &m, limb_t m_inv, const storage one, unsigned LC, unsigned LB);

    template <class storage>
    __device__ static inline int is_gt_384(const storage &left, const storage &right)
    {
        for (int i = 5; i >= 0; --i)
        {
            if (left.limbs[i] < right.limbs[i])
            {
                return 0;
            }
            else if (left.limbs[i] > right.limbs[i]) 
            {
                return 1;
            }
        }
        return 0;
    }

    template <class storage>
    __device__ static inline int is_ge_384(const storage &left, const storage &right)
    {
        for (int i = 5; i >= 0; --i)
        {
            if (left.limbs[i] < right.limbs[i])
            {
                return 0;
            }
            else if (left.limbs[i] > right.limbs[i])
            {
                return 1;
            }
        }
        return 1;
    }

    template <class storage>
    __device__ static inline void sub_384_unchecked(storage &ret, const storage &a, const storage &b)
    {
        asm(
            "sub.cc.u64 %0, %6, %12;\n\t"
            "subc.cc.u64 %1, %7, %13;\n\t"
            "subc.cc.u64 %2, %8, %14;\n\t"
            "subc.cc.u64 %3, %9, %15;\n\t"
            "subc.cc.u64 %4, %10, %16;\n\t"
            "subc.u64 %5, %11, %17;"
            : "=l"(ret.limbs[0]),
            "=l"(ret.limbs[1]),
            "=l"(ret.limbs[2]),
            "=l"(ret.limbs[3]),
            "=l"(ret.limbs[4]),
            "=l"(ret.limbs[5])
            : "l"(a.limbs[0]),
            "l"(a.limbs[1]),
            "l"(a.limbs[2]),
            "l"(a.limbs[3]),
            "l"(a.limbs[4]),
            "l"(a.limbs[5]),
            "l"(b.limbs[0]),
            "l"(b.limbs[1]),
            "l"(b.limbs[2]),
            "l"(b.limbs[3]),
            "l"(b.limbs[4]),
            "l"(b.limbs[5])
            );
    }

    template <class storage>
    __device__ static inline void reduce_384(storage &x, const storage &m)
    {
        if (is_ge_384(x, m))
        {
            storage x_sub;
            sub_384_unchecked<storage>(x_sub, x, m);
            memcpy(&x, &x_sub, sizeof(storage));
        }
    }

    // The Montgomery reduction here is based on Algorithm 14.32 in
    // Handbook of Applied Cryptography
    // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
    template <class storage>
    __device__ static inline void mont_384(storage &ret, limb_t r[12], const storage &m, const limb_t m_inv)
    {
        // printf("c-t%i:0: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);
        limb_t k = r[0] * m_inv;

        limb_t cross_carry = 0;

        asm(
            "{\n\t"
            ".reg .u64 c;\n\t"
            ".reg .u64 t;\n\t"
            ".reg .u64 nc;\n\t"

            "mad.lo.cc.u64 c, %14, %8, %0;\n\t"
            "madc.hi.cc.u64 c, %14, %8, 0;\n\t"
            
            "addc.cc.u64 t, %1, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %1, %14, %9, t;\n\t"
            "madc.hi.cc.u64 c, %14, %9, nc;\n\t"

            "addc.cc.u64 t, %2, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %2, %14, %10, t;\n\t"
            "madc.hi.cc.u64 c, %14, %10, nc;\n\t"

            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %14, %11, t;\n\t"
            "madc.hi.cc.u64 c, %14, %11, nc;\n\t"

            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %14, %12, t;\n\t"
            "madc.hi.cc.u64 c, %14, %12, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %14, %13, t;\n\t"
            "madc.hi.cc.u64 c, %14, %13, nc;\n\t"

            "addc.cc.u64 %6, %6, c;\n\t"
            "addc.u64 %7, 0, 0;\n\t"
            "}"
            : "+l"(r[0]),
            "+l"(r[1]),
            "+l"(r[2]),
            "+l"(r[3]),
            "+l"(r[4]),
            "+l"(r[5]),
            "+l"(r[6]),
            "=l"(cross_carry)
            : "l"(m.limbs[0]),
            "l"(m.limbs[1]),
            "l"(m.limbs[2]),
            "l"(m.limbs[3]),
            "l"(m.limbs[4]),
            "l"(m.limbs[5]),
            "l"(k)
        );

        // printf("c-t%i:1: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);

        k = r[1] * m_inv;

        asm(
        "{\n\t"
            ".reg .u64 c;\n\t"
            ".reg .u64 t;\n\t"
            ".reg .u64 nc;\n\t"

            "mad.lo.cc.u64 c, %14, %8, %0;\n\t"
            "madc.hi.cc.u64 c, %14, %8, 0;\n\t"
            
            "addc.cc.u64 t, %1, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %1, %14, %9, t;\n\t"
            "madc.hi.cc.u64 c, %14, %9, nc;\n\t"

            "addc.cc.u64 t, %2, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %2, %14, %10, t;\n\t"
            "madc.hi.cc.u64 c, %14, %10, nc;\n\t"

            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %14, %11, t;\n\t"
            "madc.hi.cc.u64 c, %14, %11, nc;\n\t"

            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %14, %12, t;\n\t"
            "madc.hi.cc.u64 c, %14, %12, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %14, %13, t;\n\t"
            "madc.hi.cc.u64 c, %14, %13, nc;\n\t"

            "addc.cc.u64 c, c, %7;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "addc.cc.u64 %6, %6, c;\n\t"
            "addc.u64 %7, nc, 0;\n\t"
            "}"
            : "+l"(r[1]),
            "+l"(r[2]),
            "+l"(r[3]),
            "+l"(r[4]),
            "+l"(r[5]),
            "+l"(r[6]),
            "+l"(r[7]),
            "+l"(cross_carry)
            : "l"(m.limbs[0]),
            "l"(m.limbs[1]),
            "l"(m.limbs[2]),
            "l"(m.limbs[3]),
            "l"(m.limbs[4]),
            "l"(m.limbs[5]),
            "l"(k)
        );

        // printf("c-t%i:2: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);
        k = r[2] * m_inv;

        asm(
        "{\n\t"
            ".reg .u64 c;\n\t"
            ".reg .u64 t;\n\t"
            ".reg .u64 nc;\n\t"
            
            "mad.lo.cc.u64 c, %14, %8, %0;\n\t"
            "madc.hi.cc.u64 c, %14, %8, 0;\n\t"
            
            "addc.cc.u64 t, %1, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %1, %14, %9, t;\n\t"
            "madc.hi.cc.u64 c, %14, %9, nc;\n\t"

            "addc.cc.u64 t, %2, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %2, %14, %10, t;\n\t"
            "madc.hi.cc.u64 c, %14, %10, nc;\n\t"

            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %14, %11, t;\n\t"
            "madc.hi.cc.u64 c, %14, %11, nc;\n\t"

            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %14, %12, t;\n\t"
            "madc.hi.cc.u64 c, %14, %12, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %14, %13, t;\n\t"
            "madc.hi.cc.u64 c, %14, %13, nc;\n\t"

            "addc.cc.u64 c, c, %7;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "addc.cc.u64 %6, %6, c;\n\t"
            "addc.u64 %7, nc, 0;\n\t"
            "}"
            : "+l"(r[2]),
            "+l"(r[3]),
            "+l"(r[4]),
            "+l"(r[5]),
            "+l"(r[6]),
            "+l"(r[7]),
            "+l"(r[8]),
            "+l"(cross_carry)
            : "l"(m.limbs[0]),
            "l"(m.limbs[1]),
            "l"(m.limbs[2]),
            "l"(m.limbs[3]),
            "l"(m.limbs[4]),
            "l"(m.limbs[5]),
            "l"(k)
        );

        // printf("c-t%i:3: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);

        k = r[3] * m_inv;

        asm(
        "{\n\t"
            ".reg .u64 c;\n\t"
            ".reg .u64 t;\n\t"
            ".reg .u64 nc;\n\t"
            
            "mad.lo.cc.u64 c, %14, %8, %0;\n\t"
            "madc.hi.cc.u64 c, %14, %8, 0;\n\t"
            
            "addc.cc.u64 t, %1, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %1, %14, %9, t;\n\t"
            "madc.hi.cc.u64 c, %14, %9, nc;\n\t"

            "addc.cc.u64 t, %2, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %2, %14, %10, t;\n\t"
            "madc.hi.cc.u64 c, %14, %10, nc;\n\t"

            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %14, %11, t;\n\t"
            "madc.hi.cc.u64 c, %14, %11, nc;\n\t"

            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %14, %12, t;\n\t"
            "madc.hi.cc.u64 c, %14, %12, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %14, %13, t;\n\t"
            "madc.hi.cc.u64 c, %14, %13, nc;\n\t"

            "addc.cc.u64 c, c, %7;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "addc.cc.u64 %6, %6, c;\n\t"
            "addc.u64 %7, nc, 0;\n\t"
            "}"
            : "+l"(r[3]),
            "+l"(r[4]),
            "+l"(r[5]),
            "+l"(r[6]),
            "+l"(r[7]),
            "+l"(r[8]),
            "+l"(r[9]),
            "+l"(cross_carry)
            : "l"(m.limbs[0]),
            "l"(m.limbs[1]),
            "l"(m.limbs[2]),
            "l"(m.limbs[3]),
            "l"(m.limbs[4]),
            "l"(m.limbs[5]),
            "l"(k)
        );

        // printf("c-t%i:4: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);
        k = r[4] * m_inv;

        asm(
        "{\n\t"
            ".reg .u64 c;\n\t"
            ".reg .u64 t;\n\t"
            ".reg .u64 nc;\n\t"
            
            "mad.lo.cc.u64 c, %14, %8, %0;\n\t"
            "madc.hi.cc.u64 c, %14, %8, 0;\n\t"
            
            "addc.cc.u64 t, %1, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %1, %14, %9, t;\n\t"
            "madc.hi.cc.u64 c, %14, %9, nc;\n\t"

            "addc.cc.u64 t, %2, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %2, %14, %10, t;\n\t"
            "madc.hi.cc.u64 c, %14, %10, nc;\n\t"

            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %14, %11, t;\n\t"
            "madc.hi.cc.u64 c, %14, %11, nc;\n\t"

            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %14, %12, t;\n\t"
            "madc.hi.cc.u64 c, %14, %12, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %14, %13, t;\n\t"
            "madc.hi.cc.u64 c, %14, %13, nc;\n\t"

            "addc.cc.u64 c, c, %7;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "addc.cc.u64 %6, %6, c;\n\t"
            "addc.u64 %7, nc, 0;\n\t"
            "}"
            : "+l"(r[4]),
            "+l"(r[5]),
            "+l"(r[6]),
            "+l"(r[7]),
            "+l"(r[8]),
            "+l"(r[9]),
            "+l"(r[10]),
            "+l"(cross_carry)
            : "l"(m.limbs[0]),
            "l"(m.limbs[1]),
            "l"(m.limbs[2]),
            "l"(m.limbs[3]),
            "l"(m.limbs[4]),
            "l"(m.limbs[5]),
            "l"(k)
        );

        // printf("c-t%i:5: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);
        k = r[5] * m_inv;

        asm(
        "{\n\t"
            ".reg .u64 c;\n\t"
            ".reg .u64 t;\n\t"
            ".reg .u64 nc;\n\t"
            
            "mad.lo.cc.u64 c, %14, %8, %0;\n\t"
            "madc.hi.cc.u64 c, %14, %8, 0;\n\t"
            
            "addc.cc.u64 t, %1, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %1, %14, %9, t;\n\t"
            "madc.hi.cc.u64 c, %14, %9, nc;\n\t"

            "addc.cc.u64 t, %2, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %2, %14, %10, t;\n\t"
            "madc.hi.cc.u64 c, %14, %10, nc;\n\t"

            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %14, %11, t;\n\t"
            "madc.hi.cc.u64 c, %14, %11, nc;\n\t"

            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %14, %12, t;\n\t"
            "madc.hi.cc.u64 c, %14, %12, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %14, %13, t;\n\t"
            "madc.hi.cc.u64 c, %14, %13, nc;\n\t"

            "addc.cc.u64 c, c, %7;\n\t"
            // "addc.u64 nc, 0, 0;\n\t" if we dont want to clobber cross_carry we need this
            "add.u64 %6, %6, c;\n\t" // and this to be add.cc
            // "addc.u64 %7, nc, 0;\n\t" and this
            "}"
            : "+l"(r[5]),
            "+l"(r[6]),
            "+l"(r[7]),
            "+l"(r[8]),
            "+l"(r[9]),
            "+l"(r[10]),
            "+l"(r[11])
            : "l"(cross_carry),
            "l"(m.limbs[0]),
            "l"(m.limbs[1]),
            "l"(m.limbs[2]),
            "l"(m.limbs[3]),
            "l"(m.limbs[4]),
            "l"(m.limbs[5]),
            "l"(k)
        );

        // printf("c-t%i:6: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);

        memcpy(static_cast<void*>(&ret), r + 6, sizeof(storage));

        reduce_384(ret, m);
    }

    template <class storage>
    __device__ inline void mul_mont_384(storage &ret, const storage &a, const storage &b, const storage &m, limb_t m_inv)
    {
        limb_t r[12]= {0x0};

        asm(
        "{\n\t"
            ".reg .u64 c;\n\t"
            ".reg .u64 nc;\n\t"
            ".reg .u64 t;\n\t"
            
            "mad.lo.cc.u64 %0, %12, %18, 0;\n\t"
            "madc.hi.cc.u64 c, %12, %18, 0;\n\t"
            
            "madc.lo.cc.u64 %1, %12, %19, c;\n\t"
            "madc.hi.cc.u64 c, %12, %19, 0;\n\t"

            "madc.lo.cc.u64 %2, %12, %20, c;\n\t"
            "madc.hi.cc.u64 c, %12, %20, 0;\n\t"

            "madc.lo.cc.u64 %3, %12, %21, c;\n\t"
            "madc.hi.cc.u64 c, %12, %21, 0;\n\t"

            "madc.lo.cc.u64 %4, %12, %22, c;\n\t"
            "madc.hi.cc.u64 c, %12, %22, 0;\n\t"

            "madc.lo.cc.u64 %5, %12, %23, c;\n\t"
            "madc.hi.u64 %6, %12, %23, 0;\n\t"


            "mad.lo.cc.u64 %1, %13, %18, %1;\n\t"
            "madc.hi.cc.u64 c, %13, %18, 0;\n\t"
            
            "addc.cc.u64 t, %2, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %2, %13, %19, t;\n\t"
            "madc.hi.cc.u64 c, %13, %19, nc;\n\t"

            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %13, %20, t;\n\t"
            "madc.hi.cc.u64 c, %13, %20, nc;\n\t"

            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %13, %21, t;\n\t"
            "madc.hi.cc.u64 c, %13, %21, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %13, %22, t;\n\t"
            "madc.hi.cc.u64 c, %13, %22, nc;\n\t"

            "addc.cc.u64 t, %6, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %6, %13, %23, t;\n\t"
            "madc.hi.u64 %7, %13, %23, nc;\n\t"


            "mad.lo.cc.u64 %2, %14, %18, %2;\n\t"
            "madc.hi.cc.u64 c, %14, %18, 0;\n\t"
            
            "addc.cc.u64 t, %3, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %3, %14, %19, t;\n\t"
            "madc.hi.cc.u64 c, %14, %19, nc;\n\t"
            
            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %14, %20, t;\n\t"
            "madc.hi.cc.u64 c, %14, %20, nc;\n\t"

            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %14, %21, t;\n\t"
            "madc.hi.cc.u64 c, %14, %21, nc;\n\t"

            "addc.cc.u64 t, %6, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %6, %14, %22, t;\n\t"
            "madc.hi.cc.u64 c, %14, %22, nc;\n\t"

            "addc.cc.u64 t, %7, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %7, %14, %23, t;\n\t"
            "madc.hi.u64 %8, %14, %23, nc;\n\t"



            "mad.lo.cc.u64 %3, %15, %18, %3;\n\t"
            "madc.hi.cc.u64 c, %15, %18, 0;\n\t"
            
            "addc.cc.u64 t, %4, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %4, %15, %19, t;\n\t"
            "madc.hi.cc.u64 c, %15, %19, nc;\n\t"
            
            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %15, %20, t;\n\t"
            "madc.hi.cc.u64 c, %15, %20, nc;\n\t"
            
            "addc.cc.u64 t, %6, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %6, %15, %21, t;\n\t"
            "madc.hi.cc.u64 c, %15, %21, nc;\n\t"
            
            "addc.cc.u64 t, %7, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %7, %15, %22, t;\n\t"
            "madc.hi.cc.u64 c, %15, %22, nc;\n\t"
            
            "addc.cc.u64 t, %8, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %8, %15, %23, t;\n\t"
            "madc.hi.u64 %9, %15, %23, nc;\n\t"
            



            "mad.lo.cc.u64 %4, %16, %18, %4;\n\t"
            "madc.hi.cc.u64 c, %16, %18, 0;\n\t"
            
            "addc.cc.u64 t, %5, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %5, %16, %19, t;\n\t"
            "madc.hi.cc.u64 c, %16, %19, nc;\n\t"
            
            "addc.cc.u64 t, %6, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %6, %16, %20, t;\n\t"
            "madc.hi.cc.u64 c, %16, %20, nc;\n\t"
            
            "addc.cc.u64 t, %7, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %7, %16, %21, t;\n\t"
            "madc.hi.cc.u64 c, %16, %21, nc;\n\t"
            
            "addc.cc.u64 t, %8, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %8, %16, %22, t;\n\t"
            "madc.hi.cc.u64 c, %16, %22, nc;\n\t"
            
            "addc.cc.u64 t, %9, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %9, %16, %23, t;\n\t"
            "madc.hi.u64 %10, %16, %23, nc;\n\t"
            


            "mad.lo.cc.u64 %5, %17, %18, %5;\n\t"
            "madc.hi.cc.u64 c, %17, %18, 0;\n\t"
            
            "addc.cc.u64 t, %6, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %6, %17, %19, t;\n\t"
            "madc.hi.cc.u64 c, %17, %19, nc;\n\t"
            
            "addc.cc.u64 t, %7, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %7, %17, %20, t;\n\t"
            "madc.hi.cc.u64 c, %17, %20, nc;\n\t"
            
            "addc.cc.u64 t, %8, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %8, %17, %21, t;\n\t"
            "madc.hi.cc.u64 c, %17, %21, nc;\n\t"
            
            "addc.cc.u64 t, %9, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %9, %17, %22, t;\n\t"
            "madc.hi.cc.u64 c, %17, %22, nc;\n\t"
            
            "addc.cc.u64 t, %10, c;\n\t"
            "addc.u64 nc, 0, 0;\n\t"
            "mad.lo.cc.u64 %10, %17, %23, t;\n\t"
            "madc.hi.u64 %11, %17, %23, nc;\n\t"

            "}"
            : "+l"(r[0]),
            "+l"(r[1]),
            "+l"(r[2]),
            "+l"(r[3]),
            "+l"(r[4]),
            "+l"(r[5]),
            "+l"(r[6]),
            "+l"(r[7]),
            "+l"(r[8]),
            "+l"(r[9]),
            "+l"(r[10]),
            "+l"(r[11])
            : "l"(a.limbs[0]),
            "l"(a.limbs[1]),
            "l"(a.limbs[2]),
            "l"(a.limbs[3]),
            "l"(a.limbs[4]),
            "l"(a.limbs[5]),
            "l"(b.limbs[0]),
            "l"(b.limbs[1]),
            "l"(b.limbs[2]),
            "l"(b.limbs[3]),
            "l"(b.limbs[4]),
            "l"(b.limbs[5])
        );

        mont_384(ret, r, m, m_inv);
    }

    template <class storage>
    __device__ inline void sqr_mont_384(storage &ret, const storage &a, const storage &m, limb_t m_inv) {
        limb_t r[12] = {0x0};

        asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 nc;\n\t"
        ".reg .u64 t;\n\t"

        "mad.lo.cc.u64 %1, %12, %13, 0;\n\t"
        "madc.hi.cc.u64 c, %12, %13, 0;\n\t"

        "madc.lo.cc.u64 %2, %12, %14, c;\n\t"
        "madc.hi.cc.u64 c, %12, %14, 0;\n\t"

        "madc.lo.cc.u64 %3, %12, %15, c;\n\t"
        "madc.hi.cc.u64 c, %12, %15, 0;\n\t"

        "madc.lo.cc.u64 %4, %12, %16, c;\n\t"
        "madc.hi.cc.u64 c, %12, %16, 0;\n\t"

        "madc.lo.cc.u64 %5, %12, %17, c;\n\t"
        "madc.hi.u64 %6, %12, %17, 0;\n\t"

        "mad.lo.cc.u64 %3, %13, %14, %3;\n\t"
        "madc.hi.cc.u64 c, %13, %14, 0;\n\t"
        
        "addc.cc.u64 t, %4, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %4, %13, %15, t;\n\t"
        "madc.hi.cc.u64 c, %13, %15, nc;\n\t"

        "addc.cc.u64 t, %5, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %5, %13, %16, t;\n\t"
        "madc.hi.cc.u64 c, %13, %16, nc;\n\t"

        "addc.cc.u64 t, %6, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %6, %13, %17, t;\n\t"
        "madc.hi.u64 %7, %13, %17, nc;\n\t"



        "mad.lo.cc.u64 %5, %14, %15, %5;\n\t"
        "madc.hi.cc.u64 c, %14, %15, 0;\n\t"
        
        "addc.cc.u64 t, %6, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %6, %14, %16, t;\n\t"
        "madc.hi.cc.u64 c, %14, %16, nc;\n\t"
        
        "addc.cc.u64 t, %7, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %7, %14, %17, t;\n\t"
        "madc.hi.u64 %8, %14, %17, nc;\n\t"




        "mad.lo.cc.u64 %7, %15, %16, %7;\n\t"
        "madc.hi.cc.u64 c, %15, %16, 0;\n\t"
        
        "addc.cc.u64 t, %8, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %8, %15, %17, t;\n\t"
        "madc.hi.u64 %9, %15, %17, nc;\n\t"
        


        "mad.lo.cc.u64 %9, %16, %17, %9;\n\t"
        "madc.hi.u64 %10, %16, %17, 0;\n\t"

        "}"
        : "+l"(r[0]),
        "+l"(r[1]),
        "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(r[6]),
        "+l"(r[7]),
        "+l"(r[8]),
        "+l"(r[9]),
        "+l"(r[10]),
        "+l"(r[11])
        : "l"(a.limbs[0]),
        "l"(a.limbs[1]),
        "l"(a.limbs[2]),
        "l"(a.limbs[3]),
        "l"(a.limbs[4]),
        "l"(a.limbs[5])
        );

        // printf("c-t%i:0: X, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, X\n", threadIdx.x, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]);

        r[11] = r[10] >> 63;
        r[10] = (r[10] << 1) | (r[9] >> 63);
        r[9] = (r[9] << 1) | (r[8] >> 63);
        r[8] = (r[8] << 1) | (r[7] >> 63);
        r[7] = (r[7] << 1) | (r[6] >> 63);
        r[6] = (r[6] << 1) | (r[5] >> 63);
        r[5] = (r[5] << 1) | (r[4] >> 63);
        r[4] = (r[4] << 1) | (r[3] >> 63);
        r[3] = (r[3] << 1) | (r[2] >> 63);
        r[2] = (r[2] << 1) | (r[1] >> 63);
        r[1] = r[1] << 1;

        // printf("c-t%i:1: X, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);

        asm(
        "{\n\t"

        "mad.lo.cc.u64 %0, %12, %12, 0;\n\t"
        "madc.hi.cc.u64 %1, %12, %12, %1;\n\t"

        "madc.lo.cc.u64 %2, %13, %13, %2;\n\t"
        "madc.hi.cc.u64 %3, %13, %13, %3;\n\t"
    
        "madc.lo.cc.u64 %4, %14, %14, %4;\n\t"
        "madc.hi.cc.u64 %5, %14, %14, %5;\n\t"
    
        "madc.lo.cc.u64 %6, %15, %15, %6;\n\t"
        "madc.hi.cc.u64 %7, %15, %15, %7;\n\t"
    
        "madc.lo.cc.u64 %8, %16, %16, %8;\n\t"
        "madc.hi.cc.u64 %9, %16, %16, %9;\n\t"
    
        "madc.lo.cc.u64 %10, %17, %17, %10;\n\t"
        "madc.hi.u64 %11, %17, %17, %11;\n\t"

        "}"
        : "+l"(r[0]),
        "+l"(r[1]),
        "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(r[6]),
        "+l"(r[7]),
        "+l"(r[8]),
        "+l"(r[9]),
        "+l"(r[10]),
        "+l"(r[11])
        : "l"(a.limbs[0]),
        "l"(a.limbs[1]),
        "l"(a.limbs[2]),
        "l"(a.limbs[3]),
        "l"(a.limbs[4]),
        "l"(a.limbs[5])
        );
        // printf("c-t%i:2: %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", threadIdx.x, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);

        mont_384(ret, r, m, m_inv);
    }

    template <class storage>
    __device__ static inline void add_384_unchecked(storage &ret, const storage &a, const storage &b) {


        asm(
        "add.cc.u64 %0, %6, %12;\n\t"
        "addc.cc.u64 %1, %7, %13;\n\t"
        "addc.cc.u64 %2, %8, %14;\n\t"
        "addc.cc.u64 %3, %9, %15;\n\t"
        "addc.cc.u64 %4, %10, %16;\n\t"
        "addc.u64 %5, %11, %17;"
        : "=l"(ret.limbs[0]),
        "=l"(ret.limbs[1]),
        "=l"(ret.limbs[2]),
        "=l"(ret.limbs[3]),
        "=l"(ret.limbs[4]),
        "=l"(ret.limbs[5])
        : "l"(a.limbs[0]),
        "l"(a.limbs[1]),
        "l"(a.limbs[2]),
        "l"(a.limbs[3]),
        "l"(a.limbs[4]),

        "l"(a.limbs[5]),
        "l"(b.limbs[0]),
        "l"(b.limbs[1]),
        "l"(b.limbs[2]),
        "l"(b.limbs[3]),
        "l"(b.limbs[4]),
        "l"(b.limbs[5])
        );
        // return cf != 0?
    }

    template <class storage>
    __device__ void add_mod_384(storage &ret, const storage &a, const storage &b, const storage &m)
    {
        add_384_unchecked(ret, a, b);
        reduce_384(ret, m);
    }

    template <class storage>
    __device__ void sub_mod_384(storage &ret, const storage &a, const storage &b, const storage &m)
    {
        storage added;
        memcpy(&added, static_cast<const void*>(&a), sizeof(storage));
        // printf("pre-sub [%llu, %llu, %llu, %llu, %llu, %llu]\n", added[0], added[1], added[2], added[3], added[4], added[5]);
        if (is_gt_384(b, a)) 
        {
            // printf("sub-preduce [%llu, %llu, %llu, %llu, %llu, %llu] > [%llu, %llu, %llu, %llu, %llu, %llu]\n", b[0], b[1], b[2], b[3], b[4], b[5], added[0], added[1], added[2], added[3], added[4], added[5]);
            add_384_unchecked(added, added, m);
            // printf("sub-postduce [%llu, %llu, %llu, %llu, %llu, %llu]\n", added[0], added[1], added[2], added[3], added[4], added[5]);
        }
        else
        {
        // printf("sub-nonduce [%llu, %llu, %llu, %llu, %llu, %llu] <= [%llu, %llu, %llu, %llu, %llu, %llu]\n", b[0], b[1], b[2], b[3], b[4], b[5], added[0], added[1], added[2], added[3], added[4], added[5]);
        }
        sub_384_unchecked(ret, added, b);
        // printf("post-sub [%llu, %llu, %llu, %llu, %llu, %llu]\n", ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]);
        // return cf != 0?
    }

    template <class storage>
    __device__ void sub_384_unsafe(storage &ret, const storage &a, const storage &b) {
        sub_384_unchecked(ret, a, b);
        // return cf != 0?
    }

    template <class storage>
    __device__ void add_384_unsafe(storage &ret, const storage &a, const storage &b) {
        add_384_unchecked(ret, a, b);
        // return cf != 0?
    }

    template <class storage>
    __device__ static inline void _rshift_384(storage &ret, const storage &value) {
        ret.limbs[0] = (value.limbs[1] << 63) | (value.limbs[0] >> 1);
        ret.limbs[1] = (value.limbs[2] << 63) | (value.limbs[1] >> 1);
        ret.limbs[2] = (value.limbs[3] << 63) | (value.limbs[2] >> 1);
        ret.limbs[3] = (value.limbs[4] << 63) | (value.limbs[3] >> 1);
        ret.limbs[4] = (value.limbs[5] << 63) | (value.limbs[4] >> 1);
        ret.limbs[5] = value.limbs[5] >> 1;
    }

    template <class storage>
    __device__ void div_by_2_mod_384(storage &ret, const storage &a) {
        _rshift_384(ret, a);
    }

    template <class storage>
    __device__ void cneg_mod_384(storage &ret, const storage &a, bool flag, const storage &m) {
        // just let the compiler cmov
        if (flag) {
            sub_mod_384(ret, m, a, m);
        } else {
            memcpy(ret, a, sizeof(storage));
        }
    }

    template <class storage>
    __device__ static inline int is_zero(const storage &p) {
        return p.limbs[0] == 0 &&
            p.limbs[1] == 0 &&
            p.limbs[2] == 0 &&
            p.limbs[3] == 0 &&
            p.limbs[4] == 0 &&
            p.limbs[5] == 0;
    }

    template <class storage>
    __device__ static inline int is_eq(const storage &p1, const storage &p2) {
        return p1.limbs[0] == p2.limbs[0] &&
            p1.limbs[1] == p2.limbs[1] &&
            p1.limbs[2] == p2.limbs[2] &&
            p1.limbs[3] == p2.limbs[3] &&
            p1.limbs[4] == p2.limbs[4] &&
            p1.limbs[5] == p2.limbs[5];
    }


//static inline storage inverse(storage &u, storage &v, storage &b, storage &c, const storage &m, limb_t m_inv, const storage one, unsigned LC, unsigned LB)
    template <class storage>
    __device__ storage inverse(storage &u, storage &v, storage &b, storage &c, const storage &m, limb_t m_inv, const storage one, unsigned LC, unsigned LB)
    {
        // if (is_zero(in)) {
        //     // this is really bad
        //     *((int*)NULL);
        // }
        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to
        // Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        // storage u;
        // memcpy(u, in, sizeof(storage));
        // storage v;
        // memcpy(v, BLS12_377_P, sizeof(storage));
        // storage b;
        // memcpy(b, BLS12_377_R2, sizeof(storage));
        // storage c;
        // memset(c, 0, sizeof(storage));

        while (!is_eq(u, one) && !is_eq(v, one)) {
            // printf("c-t%i-inverse_round: u=%llu v=%llu b=%llu c=%llu\n", threadIdx.x, u[0], v[0], b[0], c[0]);
            while ((u.limbs[0] & 1) == 0) {
                // printf("c-t%i-inverse_round_u_start: u=%llu b=%llu\n", threadIdx.x, u[0], b[0]);
                div_by_2_mod_384(u, u);

                if ((b.limbs[0] & 1) != 0) {
                    //blst_fp_add_unsafe(b, b, BLS12_377_P);
                    add_384_unchecked(b, b, m);
                }
                div_by_2_mod_384(b, b);
                // printf("c-t%i-inverse_round_u_stop: u=%llu b=%llu\n", threadIdx.x, u[0], b[0]);
            }

            while ((v.limbs[0] & 1) == 0) {
                // printf("c-t%i-inverse_round_v_start: u=%llu b=%llu\n", threadIdx.x, v[0], c[0]);
                div_by_2_mod_384(v, v);

                if ((c.limbs[0] & 1) != 0)
                {
                    //blst_fp_add_unsafe(c, c, BLS12_377_P);
                    add_384_unchecked(c, c, m);
                }
                div_by_2_mod_384(c, c);
                // printf("c-t%i-inverse_round_v_stop: u=%llu b=%llu\n", threadIdx.x, v[0], c[0]);
            }

            if (is_gt_384(u, v))
            {
                // blst_fp_sub_unsafe(u, u, v);
                // blst_fp_sub(b, b, c);
                sub_384_unchecked(u, u, v);
                sub_mod_384(b, b, c, m);
            } else {
                // blst_fp_sub_unsafe(v, v, u);
                // blst_fp_sub(c, c, b);
                sub_384_unchecked(v, v, u);
                sub_mod_384(c, c, b, m);
            }
        }
        if (is_eq(u, one))
        {
            return b;
        }
        else
        {
            return c;
        }
    }

    #endif
}

