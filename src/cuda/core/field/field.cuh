#pragma once

#include "../common/common.cuh"
#include "../common/math/carry_chain.cuh"
#include "../field/field_storage.cuh"
#include "../field/field_host.cuh"

using namespace common;

template <class CONFIG>
struct Field
{
    static constexpr unsigned LC = CONFIG::limbs_count;
    static constexpr unsigned BC = CONFIG::bits_count;
    static constexpr unsigned LB = CONFIG::limbs_bits;
    static constexpr unsigned BYC = CONFIG::bytes_count;

    typedef field_storage<LC> storage;
    typedef field_storage_wide<LC> storage_wide;
    storage limbs_storage;

    static constexpr HOST_DEVICE_INLINE Field get_field_zero()
    {
        Field zero = {};

        zero.limbs_storage = CONFIG::ZERO;

        return zero;
    }

    // return one in montgomery form
    static constexpr /* HOST_DEVICE_INLINE */ __host__ __device__ Field get_field_one()
    {
        Field one = {};

        one.limbs_storage = CONFIG::ONE;

        return one;
    }

    static constexpr HOST_DEVICE_INLINE storage get_one()
    {
        return CONFIG::ONE;
    }

    static constexpr HOST_DEVICE_INLINE storage get_unity_one()
    {
        return CONFIG::UNITY_ONE;
    }

    static constexpr HOST_DEVICE_INLINE storage get_R2()
    {
        return CONFIG::R2;
    }

    static constexpr HOST_DEVICE_INLINE Field get_field_R2()
    {
        Field res = {};
        res.limbs_storage = CONFIG::R2;
        return res;
    }

    static constexpr HOST_DEVICE_INLINE storage get_modular()
    {
        return CONFIG::MODULAR;
    }

    static constexpr HOST_DEVICE_INLINE Field get_field_modulus()
    {
        Field res = {};
        res.limbs_storage = CONFIG::MODULAR;
        return res;
    }

    static constexpr HOST_DEVICE_INLINE limb_t get_modular_inv()
    {
        return CONFIG::MODULAR_INV;
    }

#if 1 // panda
    template <bool SUBTRACT, bool CARRY_OUT, typename storage>
    static constexpr __host__ __device__ 
#if PANDA_ASM_32
    uint32_t 
#else
    uint64_t 
#endif
    panda_add_sub_limbs_device(const storage &xs, const storage &ys, storage &rs)
    {
#if PANDA_ASM_32
        const uint32_t *x = (uint32_t *)xs.limbs;
        const uint32_t *y = (uint32_t *)ys.limbs;
        uint32_t *r = (uint32_t *)rs.limbs;
#else
        const uint64_t *x = (uint64_t *)xs.limbs;
        const uint64_t *y = (uint64_t *)ys.limbs;
        uint64_t *r = (uint64_t *)rs.limbs;
#endif
        carry_chain<CARRY_OUT ? LC + 1 : LC> chain;

#pragma unroll
        for (unsigned i = 0; i < LC; i++)
        {
            r[i] = SUBTRACT ? chain.sub(x[i], y[i]) : chain.add(x[i], y[i]);
        }

        if (!CARRY_OUT)
        {
            return 0;
        }

        return SUBTRACT ? chain.sub(0, 0) : chain.add(0, 0);
    }

    template <typename T>
    static constexpr HOST_DEVICE_INLINE T panda_reduce(const T &xs)
    {
        const T modulus = get_modular();
        T rs = {};
        return panda_sub_limbs(xs, modulus, rs) ? xs : rs;
    }

    template <bool SUBTRACT, bool CARRY_OUT, typename T>
    static constexpr HOST_DEVICE_INLINE 
#if PANDA_ASM_32
    uint32_t 
#else
    uint64_t 
#endif 
    panda_add_sub_limbs(const T &xs, const T &ys, T &rs)
    {
#ifdef __CUDA_ARCH__
        return panda_add_sub_limbs_device<SUBTRACT, CARRY_OUT, T>(xs, ys, rs);
#else
        return 0;
#endif
    }

    static DEVICE_INLINE void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = LC)
    {
#pragma unroll
        for (size_t i = 0; i < n; i += 2)
        {
            acc[i] = ptx::mul_lo(a[i], bi);
            acc[i + 1] = ptx::mul_hi(a[i], bi);
        }
    }

    static DEVICE_INLINE void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = LC)
    {
        acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
        acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

#pragma unroll
        for (size_t i = 2; i < n; i += 2)
        {
            acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
            acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
        }
    }

    static DEVICE_INLINE void madc_n_rshift(uint32_t *odd, const uint32_t *a, uint32_t bi)
    {
        constexpr uint32_t n = LC;

#pragma unroll
        for (size_t i = 0; i < n - 2; i += 2)
        {
            odd[i] = ptx::madc_lo_cc(a[i], bi, odd[i + 2]);
            odd[i + 1] = ptx::madc_hi_cc(a[i], bi, odd[i + 3]);
        }
        odd[n - 2] = ptx::madc_lo_cc(a[n - 2], bi, 0);
        odd[n - 1] = ptx::madc_hi(a[n - 2], bi, 0);
    }

    template <typename T>
    static DEVICE_INLINE void mad_n_redc(uint32_t *even, uint32_t *odd, const uint32_t *a, uint32_t bi, bool first = false)
    {
        constexpr uint32_t n = LC;
        const T modulus = get_modular();
        const uint32_t *const MOD = (uint32_t *)modulus.limbs;
        if (first)
        {
            mul_n(odd, a + 1, bi);
            mul_n(even, a, bi);
        }
        else
        {
            even[0] = ptx::add_cc(even[0], odd[1]);
            madc_n_rshift(odd, a + 1, bi);
            cmad_n(even, a, bi);
            odd[n - 1] = ptx::addc(odd[n - 1], 0);
        }
        uint32_t mi = even[0] * get_modular_inv();
        cmad_n(odd, MOD + 1, mi);
        cmad_n(even, MOD, mi);
        odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }

    template <typename T>
    static DEVICE_INLINE void montmul_raw(const T &a_in, const T &b_in, T &r_in)
    {
        constexpr uint32_t n = LC;
        const T modulus = get_modular();
#if PANDA_ASM_32
        const uint32_t *const MOD = (uint32_t *)modulus.limbs;
        const uint32_t *a = (uint32_t *)a_in.limbs;
        const uint32_t *b = (uint32_t *)b_in.limbs;
        uint32_t *even = (uint32_t *)r_in.limbs;
        __align__(8) uint32_t odd[n + 1];
#else
        const uint64_t *const MOD = (uint64_t *)modulus.limbs;
        const uint64_t *a = (uint64_t *)a_in.limbs;
        const uint64_t *b = (uint64_t *)b_in.limbs;
        uint64_t *even = (uint64_t *)r_in.limbs;
        __align__(8) uint64_t odd[n + 1];
#endif
        int i = 0;

        for (i = 0; i < n; i += 2)
        {
            mad_n_redc<T>(&even[0], &odd[0], a, b[i], i == 0);
            mad_n_redc<T>(&odd[0], &even[0], a, b[i + 1]);
        }

#if PANDA_ASM_32
        even[0] = ptx::add_cc(even[0], odd[1]);
#else
        even[0] = ptx::u64::add_cc(even[0], odd[1]);
#endif

        for (i = 1; i < n - 1; i++)
        {
#if PANDA_ASM_32
            even[i] = ptx::addc_cc(even[i], odd[i + 1]);
#else
            even[i] = ptx::u64::addc_cc(even[i], odd[i + 1]);
#endif
        }

#if PANDA_ASM_32
        even[i] = ptx::addc(even[i], 0);
#else
        even[i] = ptx::u64::addc(even[i], 0);
#endif
        // final reduction from [0, 2*mod) to [0, mod) not done here, instead performed optionally in mul_device wrapper
    }

    template <typename T>
    static constexpr DEVICE_INLINE T panda_mul_device(const T &xs, const T &ys)
    {
        T rs = {0};

        montmul_raw<T>(xs, ys, rs);

        return panda_reduce<T>(rs);
    }

    static DEVICE_INLINE void mad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = LC)
    {
        cmad_n(odd, a + 1, bi, n - 2);
        odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, 0);
        odd[n - 1] = ptx::madc_hi(a[n - 1], bi, 0);
        cmad_n(even, a, bi, n);
        odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }

    static DEVICE_INLINE void qad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = LC)
    {
        cmad_n(odd, a, bi, n - 2);
        odd[n - 2] = ptx::madc_lo_cc(a[n - 2], bi, 0);
        odd[n - 1] = ptx::madc_hi(a[n - 2], bi, 0);
        cmad_n(even, a + 1, bi, n - 2);
        odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }

    template <typename T, typename DT>
    static DEVICE_INLINE void sqr_raw(const T &as, DT &rs)
    {
        int i = 0, j;
#if PANDA_ASM_32
        const uint32_t *a = (uint32_t *)as.limbs;
        uint32_t *even = (uint32_t *)rs.limbs;
        __align__(8) uint32_t odd[2 * LC - 2];
#else
        const uint64_t *a = (uint64_t *)as.limbs;
        uint64_t *even = (uint64_t *)rs.limbs;
        __align__(8) uint64_t odd[2 * LC - 2];
#endif
        mul_n(even + 2, a + 2, a[0], LC - 2);
        mul_n(odd, a + 1, a[0], LC);

#pragma unroll
        while (i < LC - 4)
        {
            ++i;
            mad_row(&even[2 * i + 2], &odd[2 * i], &a[i + 1], a[i], LC - i - 1);
            ++i;
            qad_row(&odd[2 * i], &even[2 * i + 2], &a[i + 1], a[i], LC - i);
        }

#if PANDA_ASM_32
        even[2 * LC - 4] = ptx::mul_lo(a[LC - 1], a[LC - 3]);
        even[2 * LC - 3] = ptx::mul_hi(a[LC - 1], a[LC - 3]);
        odd[2 * LC - 6] = ptx::mad_lo_cc(a[LC - 2], a[LC - 3], odd[2 * LC - 6]);
        odd[2 * LC - 5] = ptx::madc_hi_cc(a[LC - 2], a[LC - 3], odd[2 * LC - 5]);
        even[2 * LC - 3] = ptx::addc(even[2 * LC - 3], 0);

        odd[2 * LC - 4] = ptx::mul_lo(a[LC - 1], a[LC - 2]);
        odd[2 * LC - 3] = ptx::mul_hi(a[LC - 1], a[LC - 2]);

        even[2] = ptx::add_cc(even[2], odd[1]);

        for (j = 2; j < 2 * LC - 3; j++)
        {
            even[j + 1] = ptx::addc_cc(even[j + 1], odd[j]);
        }

        even[j + 1] = ptx::addc(odd[j], 0);

        even[0] = 0;
        even[1] = ptx::add_cc(odd[0], odd[0]);

        for (j = 2; j < 2 * LC - 1; j++)
        {
            even[j] = ptx::addc_cc(even[j], even[j]);
        }

        even[j] = ptx::addc(0, 0);

        i = 0;
        even[2 * i] = ptx::mad_lo_cc(a[i], a[i], even[2 * i]);
        even[2 * i + 1] = ptx::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
        for (++i; i < LC; i++)
        {
            even[2 * i] = ptx::madc_lo_cc(a[i], a[i], even[2 * i]);
            even[2 * i + 1] = ptx::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
        }
#else
        even[2 * LC - 4] = ptx::u64::mul_lo(a[LC - 1], a[LC - 3]);
        even[2 * LC - 3] = ptx::u64::mul_hi(a[LC - 1], a[LC - 3]);
        odd[2 * LC - 6] = ptx::u64::mad_lo_cc(a[LC - 2], a[LC - 3], odd[2 * LC - 6]);
        odd[2 * LC - 5] = ptx::u64::madc_hi_cc(a[LC - 2], a[LC - 3], odd[2 * LC - 5]);
        even[2 * LC - 3] = ptx::u64::addc(even[2 * LC - 3], 0);

        odd[2 * LC - 4] = ptx::u64::mul_lo(a[LC - 1], a[LC - 2]);
        odd[2 * LC - 3] = ptx::u64::mul_hi(a[LC - 1], a[LC - 2]);

        // merge |even[2:]| and |odd[1:]|
        even[2] = ptx::u64::add_cc(even[2], odd[1]);

        for (j = 2; j < 2 * LC - 3; j++)
        {
            even[j + 1] = ptx::u64::addc_cc(even[j + 1], odd[j]);
        }

        even[j + 1] = ptx::u64::addc(odd[j], 0);

        // double |even|
        even[0] = 0;
        even[1] = ptx::u64::add_cc(odd[0], odd[0]);

        for (j = 2; j < 2 * LC - 1; j++)
        {
            even[j] = ptx::u64::addc_cc(even[j], even[j]);
        }

        even[j] = ptx::u64::addc(0, 0);

        // accumulate "diagonal" |a[i]|*|a[i]| product
        i = 0;
        even[2 * i] = ptx::u64::mad_lo_cc(a[i], a[i], even[2 * i]);
        even[2 * i + 1] = ptx::u64::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
        for (++i; i < LC - 1; i++)
        {
            even[2 * i] = ptx::u64::madc_lo_cc(a[i], a[i], even[2 * i]);
            even[2 * i + 1] = ptx::u64::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
        }
        even[2 * i] = ptx::u64::madc_lo_cc(a[i], a[i], even[2 * i]);
        even[2 * i + 1] = ptx::u64::madc_hi(a[i], a[i], even[2 * i + 1]);
#endif
    }

    template <typename T>
    static DEVICE_INLINE void mul_by_1_row(uint32_t *even, uint32_t *odd, bool first = false)
    {
        uint32_t mi;
        constexpr T modulus = get_modular();
        const uint32_t *const MOD = (uint32_t *)modulus.limbs;

        if (first)
        {
            mi = even[0] * get_modular_inv();
            mul_n(odd, MOD + 1, mi);
            cmad_n(even, MOD, mi);
            odd[LC - 1] = ptx::addc(odd[LC - 1], 0);
        }
        else
        {
            even[0] = ptx::add_cc(even[0], odd[1]);
            mi = even[0] * get_modular_inv();
            madc_n_rshift(odd, MOD + 1, mi);
            cmad_n(even, MOD, mi);
            odd[LC - 1] = ptx::addc(odd[LC - 1], 0);
        }
    }

    template <typename T, typename DT>
    static DEVICE_INLINE void redc_wide_inplace(DT &xs)
    {
#if PANDA_ASM_32
        uint32_t *even = (uint32_t *)xs.limbs;

        uint32_t odd[LC];
        size_t i;

#pragma unroll
        for (i = 0; i < LC; i += 2)
        {
            mul_by_1_row<T>(&even[0], &odd[0], i == 0);
            mul_by_1_row<T>(&odd[0], &even[0]);
        }
        even[0] = ptx::add_cc(even[0], odd[1]);

#pragma unroll
        for (i = 1; i < LC - 1; i++)
        {
            even[i] = ptx::addc_cc(even[i], odd[i + 1]);
        }

        even[i] = ptx::addc(even[i], 0);

        xs.limbs[0] = ptx::add_cc(xs.limbs[0], xs.limbs[LC]);

#pragma unroll
        for (i = 1; i < LC - 1; i++)
        {
            xs.limbs[i] = ptx::addc_cc(xs.limbs[i], xs.limbs[i + LC]);
        }

        xs.limbs[LC - 1] = ptx::addc(xs.limbs[LC - 1], xs.limbs[2 * LC - 1]);
#else
        uint64_t *even = (uint64_t *)xs.limbs;
        // Yields montmul of lo LC limbs * 1.
        // Since the hi LC limbs don't participate in computing the "mi" factor at each mul-and-rightshift stage,
        // it's ok to ignore the hi LC limbs during this process and just add them in afterward.
        uint64_t odd[LC]{0x0};
        size_t i;

#pragma unroll
        for (i = 0; i < LC; i += 2)
        {
            mul_by_1_row<T>(&even[0], &odd[0], i == 0);
            mul_by_1_row<T>(&odd[0], &even[0]);
        }
        even[0] = ptx::u64::add_cc(even[0], odd[1]);

#pragma unroll
        for (i = 1; i < LC - 1; i++)
        {
            even[i] = ptx::u64::addc_cc(even[i], odd[i + 1]);
        }

        even[i] = ptx::u64::addc(even[i], 0);
        // Adds in (hi LC limbs), implicitly right-shifting them by LC limbs as if they had participated in the
        // add-and-rightshift stages above.
        xs.limbs[0] = ptx::u64::add_cc(xs.limbs[0], xs.limbs[LC]);

#pragma unroll
        for (i = 1; i < LC - 1; i++)
        {
            xs.limbs[i] = ptx::u64::addc_cc(xs.limbs[i], xs.limbs[i + LC]);
        }

        xs.limbs[LC - 1] = ptx::u64::addc(xs.limbs[LC - 1], xs.limbs[2 * LC - 1]);
#endif
    }

    template <typename T, typename DT>
    static constexpr DEVICE_INLINE T panda_sqr_device(const T &xs)
    {
        DT rs = {0};
        sqr_raw<T, DT>(xs, rs);
        redc_wide_inplace<T, DT>(rs);
        return panda_reduce(rs.get_lo());
    }

    template <typename T>
    static constexpr HOST_DEVICE_INLINE uint32_t panda_add_limbs(const T &xs, const T &ys, T &rs)
    {
        return panda_add_sub_limbs<false, false>(xs, ys, rs);
    }

    template <bool CARRY_OUT, typename T>
    static constexpr HOST_DEVICE_INLINE uint32_t panda_sub_limbs(const T &xs, const T &ys, T &rs)
    {
        return panda_add_sub_limbs<true, CARRY_OUT>(xs, ys, rs);
    }

    template <typename T>
    static constexpr HOST_DEVICE_INLINE 
#if PANDA_ASM_32
    uint32_t 
#else
    uint64_t 
#endif
    panda_sub_limbs(const T &xs, const T &ys, T &rs)
    {
        return panda_add_sub_limbs<true, true>(xs, ys, rs);
    }

    template <typename T>
    static constexpr HOST_DEVICE_INLINE T panda_mul_limbs(const T &xs, const T &ys)
    {
        return panda_mul_device(xs, ys);
    }

    template <typename T, typename DT>
    static constexpr HOST_DEVICE_INLINE T panda_sqr_limbs(const T &xs)
    {
        return panda_sqr_device<T, DT>(xs);
    }

    template <typename T>
    static constexpr HOST_DEVICE_INLINE T panda_add_operator(const T &xs, const T &ys)
    {
        T rs;
        panda_add_limbs(xs, ys, rs);

        return panda_reduce(rs);
    }

    template <typename T>
    static constexpr HOST_DEVICE_INLINE T panda_sub_operator(const T &xs, const T &ys)
    {
        T rs;
#if PANDA_ASM_32
    uint32_t carry = panda_sub_limbs(xs, ys, rs);
#else
    uint64_t carry = panda_sub_limbs(xs, ys, rs);
#endif

        if (carry == 0)
        {
            return rs;
        }

        const T modulus = get_modular();
        panda_add_limbs(rs, modulus, rs);
        return rs;
    }

    template <typename T>
    static constexpr HOST_DEVICE_INLINE T panda_mul_operator(const T &xs, const T &ys)
    {
        return panda_mul_limbs(xs, ys);
    }

    template <typename T, typename DT>
    static constexpr HOST_DEVICE_INLINE T panda_sqr_operator(const T &xs)
    {
        return panda_sqr_limbs<T, DT>(xs);
    }
#endif

    static constexpr HOST_DEVICE_INLINE Field to_montgomery(const Field &x)
    {
        constexpr storage R2 = get_R2();
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();
        Field res = {};

#ifdef __CUDA_ARCH__

#if PANDA_ASM
        res.limbs_storage = panda_mul_operator<storage>(x.limbs_storage, R2);
#else
        if (x.LC == 6)
        {
            ASM_CUDA_FP384::mul_mont_384<storage>(res.limbs_storage, x.limbs_storage, R2, MODULAR, MODULAR_INV);
        }
        else if (x.LC == 4)
        {
            ASM_CUDA_FP256::mul_mont_256<storage>(res.limbs_storage, x.limbs_storage, R2, MODULAR, MODULAR_INV);
        }
#endif
#else
        FIELD_HOST::mul_mod_limbs_host<storage, storage_wide>(res.limbs_storage, x.limbs_storage, R2, MODULAR, MODULAR_INV, LC);
#endif

        return res;
    }

    static constexpr HOST_DEVICE_INLINE Field from_montgomery(const Field &x)
    {
        constexpr storage UNITY_ONE = get_unity_one();
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();
        Field res = {};

#ifdef __CUDA_ARCH__
#if PANDA_ASM
        res.limbs_storage = panda_mul_operator<storage>(x.limbs_storage, UNITY_ONE);
#else
        if (x.LC == 6)
        {
            ASM_CUDA_FP384::mul_mont_384<storage>(res.limbs_storage, x.limbs_storage, UNITY_ONE, MODULAR, MODULAR_INV);
        }
        else if (x.LC == 4)
        {
            ASM_CUDA_FP256::mul_mont_256<storage>(res.limbs_storage, x.limbs_storage, UNITY_ONE, MODULAR, MODULAR_INV);
        }
#endif
#else
        FIELD_HOST::mul_mod_limbs_host<storage, storage_wide>(res.limbs_storage, x.limbs_storage, UNITY_ONE, MODULAR, MODULAR_INV, LC);
#endif

        return res;
    }

    static constexpr HOST_DEVICE_INLINE void add_mod_limbs(Field &res, const Field &x, const Field &y)
    {
        const storage MODULAR = get_modular();

#ifdef __CUDA_ARCH__
#if PANDA_ASM
        res.limbs_storage = panda_add_operator<storage>(x.limbs_storage, y.limbs_storage);
#else
        ASM_CUDA_FP384::add_mod_384<storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR);
#endif
#else
        FIELD_HOST::add_mod_limbs_host<storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, LC);
#endif
    }

    static constexpr HOST_DEVICE_INLINE void sub_mod_limbs(Field &res, const Field &x, const Field &y)
    {
#ifdef __CUDA_ARCH__
#if PANDA_ASM
        res.limbs_storage = panda_sub_operator<storage>(x.limbs_storage, y.limbs_storage);
#else
        const storage MODULAR = get_modular();
        ASM_CUDA_FP384::sub_mod_384<storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR);
#endif
#else
        const storage MODULAR = get_modular();
        FIELD_HOST::sub_mod_limbs_host<storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, LC);
#endif
    }

    static constexpr HOST_DEVICE_INLINE void mul_mod_limbs(Field &res, const Field &x, const Field &y)
    {
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();

#ifdef __CUDA_ARCH__
#if PANDA_ASM
            res.limbs_storage = panda_mul_operator<storage>(x.limbs_storage, y.limbs_storage);
#else
        if (x.LC == 6)
        {
            ASM_CUDA_FP384::mul_mont_384<storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, MODULAR_INV);
        }
        else if (x.LC == 4)
        {
            ASM_CUDA_FP256::mul_mont_256<storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, MODULAR_INV);
        }
#endif
#else
        FIELD_HOST::mul_mod_limbs_host<storage, storage_wide>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, MODULAR_INV, LC);
#endif
    }

    static constexpr HOST_DEVICE_INLINE void sqr_mod_limbs(Field &res, const Field &x)
    {
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();

#ifdef __CUDA_ARCH__
#if PANDA_ASM
        res.limbs_storage = panda_sqr_operator<storage, storage_wide>(x.limbs_storage);
#else
        ASM_CUDA_FP384::sqr_mont_384<storage>(res.limbs_storage, x.limbs_storage, MODULAR, MODULAR_INV);
#endif
#else
        FIELD_HOST::sqr_mod_limbs_host<storage, storage_wide>(res.limbs_storage, x.limbs_storage, MODULAR, MODULAR_INV, LC);
#endif
    }

    static constexpr HOST_DEVICE_INLINE Field neg(const Field &x)
    {
        const storage MODULAR = get_modular();
        Field res = {};
        Field modular = {};
        modular.limbs_storage = MODULAR;

        sub_mod_limbs(res, modular, x);

        return res;
    }

    static constexpr HOST_DEVICE_INLINE bool is_zero(const Field &x)
    {
#pragma unroll
        for (int i = 0; i < LC; i++)
        {
            if (x.limbs_storage.limbs[i])
            {
                return false;
            }
        }

        return true;
    }

    static constexpr HOST_DEVICE_INLINE bool is_one(const Field &x)
    {
        for (unsigned i = 0; i < LC; i++)
        {
            if (i == 0)
            {
                if (x.limbs_storage.limbs[0] == 0x01)
                {
                    continue;
                }
                else
                {
                    return false;
                }
            }

            if (x.limbs_storage.limbs[i] != 0x0)
            {
                return false;
            }
        }

        return true;
    }

    friend HOST_DEVICE_INLINE Field operator+(Field x, const Field &y)
    {
        Field res = {};
        add_mod_limbs(res, x, y);
        return res;
    }

    HOST_DEVICE_INLINE Field& operator+=(const storage &y)
    {
        panda_add_limbs<storage>(this->limbs_storage, y, this->limbs_storage);
        return *this;
    }

    friend HOST_DEVICE_INLINE Field operator-(Field x, const Field &y)
    {
        Field res = {};
        sub_mod_limbs(res, x, y);

        return res;
    }

    HOST_DEVICE_INLINE Field& operator-=(const Field &y)
    {
        panda_sub_limbs<false>(this->limbs_storage, y.limbs_storage, this->limbs_storage);
        return *this;
    }

    friend HOST_DEVICE_INLINE Field operator*(const Field &x, const Field &y)
    {
        Field res = {};
        mul_mod_limbs(res, x, y);

        return res;
    }

    friend HOST_DEVICE_INLINE Field operator^(const Field &x, int exponent)
    {
        Field res = {};
        if (exponent == 2)
        {
            sqr_mod_limbs(res, x);
        }
        else
        {
            // Handle other cases (optional)
        }

        return res;
    }

    friend HOST_DEVICE_INLINE bool operator==(const Field &x, const Field &y)
    {
        for (int i = 0; i < LC; i++)
        {
            if (x.limbs_storage.limbs[i] != y.limbs_storage.limbs[i])
            {
                return false;
            }
        }

        return true;
    }

    friend HOST_DEVICE_INLINE bool operator!=(const Field &x, const Field &y)
    {
        for (int i = 0; i < LC; i++)
        {
            if (x.limbs_storage.limbs[i] != y.limbs_storage.limbs[i])
            {
                return true;
            }
        }

        return false;
    }

    friend HOST_DEVICE_INLINE bool operator>(const Field &left, const Field &right)
    {
        for (int i = LC - 1; i >= 0; --i)
        {
            if (left.limbs_storage.limbs[i] < right.limbs_storage.limbs[i])
            {
                return false;
            }
            else if (left.limbs_storage.limbs[i] > right.limbs_storage.limbs[i])
            {
                return true;
            }
        }

        return false;
    }

#if 0      
    static constexpr HOST_DEVICE_INLINE Field mul_scalar(const unsigned scalar, const Field &x)
    {
        Field res = {};
        Field temp = x;
        unsigned l = scalar;
        bool is_zero = true;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (unsigned i = 0; i < 32; i++)
        {
            if (l & 1)
            {
                // res = is_zero ? temp : (l >> 1) ? add(res, temp) : add(res, temp);
                if (is_zero == 0)
                {
                    res = temp;
                }
                else
                {
                    dbl_mod_limbs(res, res, temp);
                }
                is_zero = false;
            }
            l >>= 1;
            if (l == 0)
            {
                break;
            }

            temp = dbl<REDUCTION_SIZE>(temp);
        }
        return res;
    }
#endif

    // from montgomery to bigint
    static constexpr HOST_DEVICE_INLINE Field to_bigint(const Field &input)
    {
        return from_montgomery(input);
    }

    // from bigint to montgomery
    static constexpr HOST_DEVICE_INLINE Field from_bigint(const Field &input)
    {
        return to_montgomery(input);
    }

    static constexpr HOST_DEVICE_INLINE Field read_le(const Field &input)
    {
        return from_bigint(input);
    }

    static constexpr HOST_DEVICE_INLINE Field write_le(const Field &input)
    {
        return to_bigint(input);
    }

    static constexpr HOST_DEVICE_INLINE Field deserialize_uncompressed(Field &input)
    {
        return read_le(input);
    }

    static Field inverse(const Field &x)
    {
        Field u = x;
        Field v = get_field_modulus();
        Field b = get_field_R2();
        Field c = get_field_zero();
        // Field one = get_field_one();
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();
        constexpr storage UNITY_ONE = get_unity_one();

        Field res = {};
        res.limbs_storage = FIELD_HOST::inverse<storage>(u.limbs_storage, v.limbs_storage, b.limbs_storage, c.limbs_storage, MODULAR, MODULAR_INV, UNITY_ONE, LC, LB);

        return res;
    }

    static __host__ __device__ void div2(storage &xs)
    {
        for (unsigned i = 0; i < LC - 1; i++)
        {
            xs.limbs[i] = (xs.limbs[i] >> 1) | (xs.limbs[i + 1] << (FIELD_BITS - 1));
        }

        xs.limbs[LC - 1] = xs.limbs[LC - 1] >> 1;
    }

    static __host__ __device__ Field panda_inverse(const Field &x)
    {
        if (is_zero(x))
        {
            return x;
        }

        Field u = x;
        Field v = get_field_modulus();
        Field b = get_field_R2();
        Field c = get_field_zero();

        const storage MODULAR = get_modular();
        // const storage UNITY_ONE = get_unity_one();
        const Field UNITY_ONE {get_unity_one()};

        while (u != UNITY_ONE && v != UNITY_ONE)
        {
            while (!(u.limbs_storage.limbs[0] & 0x1))
            {
                div2(u.limbs_storage);
                if (b.limbs_storage.limbs[0] & 1)
                    b += MODULAR;
                div2(b.limbs_storage);
            }

            while (!(v.limbs_storage.limbs[0] & 0x1))
            {
                div2(v.limbs_storage);
                if (c.limbs_storage.limbs[0] & 1)
                    c += MODULAR;
                div2(c.limbs_storage);
            }

            if (u > v)
            {
                u -= v;
                b = b - c;
            }
            else
            {
                v -= u;
                c = c - b;
            }
        }

        return u == UNITY_ONE ? b : c;
    }

    static constexpr HOST_DEVICE_INLINE int cmp_bigint(const Field &xR, const Field &yR)
    {
        Field x = to_bigint(xR);
        Field y = to_bigint(yR);

        return x > y;
    }

    template <unsigned MODULUS_SIZE = 1> friend HOST_DEVICE_INLINE void neg(const Field& var, Field& output)
    {
        Field modulus = get_field_modulus();

        panda_sub_limbs<false>(modulus.limbs_storage, var.limbs_storage, output.limbs_storage);
    }

    static constexpr HOST_DEVICE_INLINE void print(const Field &x)
    {
        printf("\t0x");
        for (int i = LC - 1; i >= 0; i--)
        {

#if PANDA_ASM_32
            printf("%08x", x.limbs_storage.limbs[i]);
#elif PANDA_ASM_64
            printf("%016llx", x.limbs_storage.limbs[i]);
#else
            printf("%016llx", x.limbs_storage.limbs[i]);
#endif

            if (i != 0)
            {
                printf("_");
            }
        }
        printf("\n");
    }

    static constexpr HOST_DEVICE_INLINE void print(const char *info, const Field &x)
    {
        printf("%s\t0x", info);
        for (int i = LC - 1; i >= 0; i--)
        {
#if PANDA_ASM_32
            printf("%08x", x.limbs_storage.limbs[i]);
#elif PANDA_ASM_64
            printf("%016llx", x.limbs_storage.limbs[i]);
#else
            printf("%016llx", x.limbs_storage.limbs[i]);
#endif

            if (i != 0)
            {
                printf("_");
            }
        }
        printf("\n");
    }
};