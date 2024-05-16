#pragma once

#include "field_storage.cuh"
#include "field.cuh"
#include "common/common.cuh"
#include "common/math/math_64bit.cuh"
#include "common/math/math_32bit.cuh"

using namespace common;

namespace FIELD_HOST
{
    // l >= r is 1
    template <class storage>
    inline int is_ge_limbs(const storage &left, const storage &right, unsigned LC)
    {
        for (int i = (LC - 1); i >= 0; --i)
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

    // l > r is 1
    template <class storage>
    inline int is_gt_limbs(const storage &left, const storage &right, unsigned LC)
    {
        for (int i = (LC - 1); i >= 0; --i)
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
    inline int is_equal(const storage &left, const storage &right, unsigned LC)
    {
        for (int i = (LC - 1); i >= 0; --i)
        {
            if (left.limbs[i] != right.limbs[i])
            {
                return 0;
            }
        }

        return 1;
    }

    template <class storage>
    inline void add_mod_limbs_unchecked(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        limb_t carry = 0x0;
#if PANDA_ASM
#if PANDA_ASM_32
        uint32_t carry_value = static_cast<uint32_t>(carry);
#elif PANDA_ASM_64
        uint64_t carry_value = static_cast<uint64_t>(carry);
#endif
#else
        uint64_t carry_value = static_cast<uint64_t>(carry);
#endif

        for (unsigned i = 0; i < LC; i++)
        {
#if PANDA_ASM
#if PANDA_ASM_32
            res.limbs[i] = MATH_32BIT::add_with_carry_in((uint32_t)x.limbs[i], (uint32_t)y.limbs[i], &carry_value);
#elif PANDA_ASM_64
            res.limbs[i] = MATH_64BIT::add_with_carry_in((uint64_t)x.limbs[i], (uint64_t)y.limbs[i], &carry_value);
#endif
#else
            res.limbs[i] = MATH_64BIT::add_with_carry_in((uint64_t)x.limbs[i], (uint64_t)y.limbs[i], &carry_value);
#endif
        }
    }

    template <class storage>
    static inline void sub_mod_limbs_unchecked(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        limb_t borrow = 0x0;
#if PANDA_ASM
#if PANDA_ASM_32
        uint32_t borrow_value = static_cast<uint32_t>(borrow);
#elif PANDA_ASM_64
        uint64_t borrow_value = static_cast<uint64_t>(borrow);
#endif
#else
        uint64_t borrow_value = static_cast<uint64_t>(borrow);
#endif

        for (unsigned i = 0; i < LC; i++)
        {
#if PANDA_ASM
#if PANDA_ASM_32
            res.limbs[i] = MATH_32BIT::sub_with_borrow((uint32_t)x.limbs[i], (uint32_t)y.limbs[i], &borrow_value);
#elif PANDA_ASM_64
            res.limbs[i] = MATH_64BIT::sub_with_borrow((uint64_t)x.limbs[i], (uint64_t)y.limbs[i], &borrow_value);
#endif
#else
            res.limbs[i] = MATH_64BIT::sub_with_borrow((uint64_t)x.limbs[i], (uint64_t)y.limbs[i], &borrow_value);
#endif
        }
    }

    template <class storage>
    static inline void reduce_limbs(storage &x, const storage &m, unsigned LC)
    {
        if (is_ge_limbs(x, m, LC))
        {
            storage x_sub = {};
            sub_mod_limbs_unchecked(x_sub, x, m, LC);
            memcpy(&x, &x_sub, sizeof(storage));
        }
    }

    template <class storage>
    static constexpr HOST_INLINE void add_mod_limbs_host(storage &res, const storage &x, const storage &y, const storage &m, unsigned LC)
    {
        add_mod_limbs_unchecked(res, x, y, LC);
        reduce_limbs(res, m, LC);
    }

    template <class storage>
    static constexpr HOST_INLINE void add_limbs_host(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        add_mod_limbs_unchecked(res, x, y, LC);
    }

    template <class storage>
    static constexpr HOST_INLINE void sub_mod_limbs_host(storage &res, const storage &x, const storage &y, const storage &m, unsigned LC)
    {
        storage added = x;

        if (is_ge_limbs(y, x, LC))
        {
            add_mod_limbs_unchecked(added, added, m, LC);
        }

        sub_mod_limbs_unchecked(res, added, y, LC);
    }

    template <class storage>
    static constexpr HOST_INLINE void sub_limbs_host(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        sub_mod_limbs_unchecked(res, x, y, LC);
    }

    template <class storage, class storage_wide>
    inline void mul_limbs(storage_wide &res, const storage &x, const storage &y, unsigned LC)
    {

        limb_t carry = 0x0;
#if PANDA_ASM
#if PANDA_ASM_32
        uint32_t carry_value = static_cast<uint32_t>(carry);
#elif PANDA_ASM_64
        uint64_t carry_value = static_cast<uint64_t>(carry);
#endif
#else
        uint64_t carry_value = static_cast<uint64_t>(carry);
#endif

        for (unsigned i = 0; i < LC; i++)
        {
            carry_value = 0x0;
            for (unsigned j = 0; j < LC; j++)
            {
                if (i == 0 && j == 0)
                {
// first operation in round 1
#if PANDA_ASM
#if PANDA_ASM_32
                    res.limbs[i + j] = MATH_32BIT::mul_with_carry((uint32_t)x.limbs[i + j], (uint32_t)y.limbs[i + j], &carry_value);
#elif PANDA_ASM_64
                    res.limbs[i + j] = MATH_64BIT::mul_with_carry((uint64_t)x.limbs[i + j], (uint64_t)y.limbs[i + j], &carry_value);
#endif
#else
                    res.limbs[i + j] = MATH_64BIT::mul_with_carry((uint64_t)x.limbs[i + j], (uint64_t)y.limbs[i + j], &carry_value);
#endif
                }
                else
                {
#if PANDA_ASM
#if PANDA_ASM_32
                    res.limbs[i + j] = MATH_32BIT::mac_with_carry_in((uint32_t)x.limbs[i], (uint32_t)y.limbs[j], (uint32_t)res.limbs[i + j], &carry_value);
#elif PANDA_ASM_64
                    res.limbs[i + j] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i], (uint64_t)y.limbs[j], (uint64_t)res.limbs[i + j], &carry_value);
#endif
#else
                    res.limbs[i + j] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i], (uint64_t)y.limbs[j], (uint64_t)res.limbs[i + j], &carry_value);
#endif
                }
            }
            res.limbs[i + LC] = carry_value;
        }
    }

    template <class storage, class storage_wide>
    void sqr_limbs(storage_wide &res, const storage &x, unsigned LC)
    {
        limb_t carry = 0x0;
#if PANDA_ASM
#if PANDA_ASM_32
        uint32_t carry_value = static_cast<uint32_t>(carry);
#elif PANDA_ASM_64
        uint64_t carry_value = static_cast<uint64_t>(carry);
#endif
#else
        uint64_t carry_value = static_cast<uint64_t>(carry);
#endif

        for (unsigned i = 0; i < LC - 1; i++)
        {
            carry_value = 0x0;
            for (unsigned j = 0; j < LC - 1 - i; j++)
            {
                if (i == 0 && j == 0)
                {
#if PANDA_ASM
#if PANDA_ASM_32
                    res.limbs[1] = MATH_32BIT::mul_with_carry((uint32_t)x.limbs[0], (uint32_t)x.limbs[1], &carry_value);
#elif PANDA_ASM_64
                    res.limbs[1] = MATH_64BIT::mul_with_carry((uint64_t)x.limbs[0], (uint64_t)x.limbs[1], &carry_value);
#endif
#else
                    res.limbs[1] = MATH_64BIT::mul_with_carry((uint64_t)x.limbs[0], (uint64_t)x.limbs[1], &carry_value);
#endif
                }
                else
                {
#if PANDA_ASM
#if PANDA_ASM_32
                    res.limbs[i * 2 + j + 1] = MATH_32BIT::mac_with_carry_in((uint32_t)x.limbs[i], (uint32_t)x.limbs[i + j + 1], (uint32_t)res.limbs[i * 2 + j + 1], &carry_value);
#elif PANDA_ASM_64
                    res.limbs[i * 2 + j + 1] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i], (uint64_t)x.limbs[i + j + 1], (uint64_t)res.limbs[i * 2 + j + 1], &carry_value);
#endif
#else
                    res.limbs[i * 2 + j + 1] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i], (uint64_t)x.limbs[i + j + 1], (uint64_t)res.limbs[i * 2 + j + 1], &carry_value);
#endif
                }
            }
            res.limbs[i + 1 + LC - 1] = carry_value;
        }

        for (unsigned i = 0; i < 2 * LC - 1; i++)
        {
            // note
            unsigned rshift_count = 8 * sizeof(limb_t) - 1;

            if (i == 0)
            {
                res.limbs[2 * LC - 1] = res.limbs[2 * LC - 2] >> rshift_count;
            }
            else if (i == 2 * LC - 2)
            {
                res.limbs[2 * LC - 1 - i] = (res.limbs[2 * LC - 1 - i] << 1);
            }
            else
            {
                res.limbs[2 * LC - 1 - i] = (res.limbs[2 * LC - 1 - i] << 1) | (res.limbs[2 * LC - 2 - i] >> rshift_count);
            }
        }

        carry_value = 0x0;
        for (unsigned i = 0; i < 2 * LC; i++)
        {
            if (i % 2 == 0)
            {
#if PANDA_ASM
#if PANDA_ASM_32
                res.limbs[i] = MATH_32BIT::mac_with_carry_in((uint32_t)x.limbs[i >> 1], (uint32_t)x.limbs[i >> 1], (uint32_t)res.limbs[i], &carry_value);
#elif PANDA_ASM_64
                res.limbs[i] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i >> 1], (uint64_t)x.limbs[i >> 1], (uint64_t)res.limbs[i], &carry_value);
#endif
#else
                res.limbs[i] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i >> 1], (uint64_t)x.limbs[i >> 1], (uint64_t)res.limbs[i], &carry_value);
#endif
            }
            else
            {
#if PANDA_ASM
#if PANDA_ASM_32
                res.limbs[i] = MATH_32BIT::add_carry_with_carry((uint32_t)res.limbs[i], &carry_value);
#elif PANDA_ASM_64
                res.limbs[i] = MATH_64BIT::add_carry_with_carry((uint64_t)res.limbs[i], &carry_value);
#endif
#else
                res.limbs[i] = MATH_64BIT::add_carry_with_carry((uint64_t)res.limbs[i], &carry_value);
#endif
            }
        }
    }

    template <class storage, class storage_wide>
    inline void mont_limbs(storage &res, storage_wide &r, const storage &m, limb_t m_inv, unsigned LC) //(blst_fp ret.limbs, blst_fp_double r, const blst_fp p, uint64_t p_inv)
    {
        limb_t carry2 = 0;

#if PANDA_ASM
#if PANDA_ASM_32
        uint32_t carry2_value = static_cast<uint32_t>(carry2);
#elif PANDA_ASM_64
        uint64_t carry2_value = static_cast<uint64_t>(carry2);
#endif
#else
        uint64_t carry2_value = static_cast<uint64_t>(carry2);
#endif

        for (unsigned i = 0; i < LC; i++)
        {
            limb_t k = r.limbs[i] * m_inv;
            limb_t carry1 = 0;
#if PANDA_ASM
#if PANDA_ASM_32
            uint32_t carry1_value = static_cast<uint32_t>(carry1);
#elif PANDA_ASM_64
            uint64_t carry1_value = static_cast<uint64_t>(carry1);
#endif
#else
            uint64_t carry1_value = static_cast<uint64_t>(carry1);
#endif

            for (unsigned j = 0; j < LC; j++)
            {
                if (j == 0)
                {
#if PANDA_ASM
#if PANDA_ASM_32
                    MATH_32BIT::mac_with_carry_in((uint32_t)k, (uint32_t)m.limbs[j], (uint32_t)r.limbs[i + j], &carry1_value);
#elif PANDA_ASM_64
                    MATH_64BIT::mac_with_carry_in((uint64_t)k, (uint64_t)m.limbs[j], (uint64_t)r.limbs[i + j], &carry1_value);
#endif
#else
                    MATH_64BIT::mac_with_carry_in((uint64_t)k, (uint64_t)m.limbs[j], (uint64_t)r.limbs[i + j], &carry1_value);
#endif
                }
                else
                {
#if PANDA_ASM
#if PANDA_ASM_32
                    r.limbs[i + j] = MATH_32BIT::mac_with_carry_in((uint32_t)k, (uint32_t)m.limbs[j], (uint32_t)r.limbs[i + j], &carry1_value);
#elif PANDA_ASM_64
                    r.limbs[i + j] = MATH_64BIT::mac_with_carry_in((uint64_t)k, (uint64_t)m.limbs[j], (uint64_t)r.limbs[i + j], &carry1_value);
#endif
#else
                    r.limbs[i + j] = MATH_64BIT::mac_with_carry_in((uint64_t)k, (uint64_t)m.limbs[j], (uint64_t)r.limbs[i + j], &carry1_value);
#endif
                }
            }
#if PANDA_ASM
#if PANDA_ASM_32
            r.limbs[i + LC] = MATH_32BIT::add_with_carry_in((uint32_t)r.limbs[i + LC], carry2_value, &carry1_value);
#elif PANDA_ASM_64
            r.limbs[i + LC] = MATH_64BIT::add_with_carry_in((uint64_t)r.limbs[i + LC], carry2_value, &carry1_value);
#endif
#else
            r.limbs[i + LC] = MATH_64BIT::add_with_carry_in((uint64_t)r.limbs[i + LC], carry2_value, &carry1_value);
#endif

            carry2_value = carry1_value;
        }

        for (unsigned i = 0; i < LC; i++)
        {
            res.limbs[i] = r.limbs[i + LC];
        }
        reduce_limbs(res, m, LC);
    }

    template <class storage, class storage_wide>
    static constexpr HOST_INLINE void mul_mod_limbs_host(storage &res, const storage &x, const storage &y, const storage &m, limb_t m_inv, unsigned LC)
    {
        storage_wide r = {};

        mul_limbs(r, x, y, LC);

        mont_limbs(res, r, m, m_inv, LC);
    }

    template <class storage, class storage_wide>
    static constexpr HOST_INLINE void sqr_mod_limbs_host(storage &res, const storage &x, const storage &m, limb_t m_inv, unsigned LC)
    {
        storage_wide r = {};

        sqr_limbs(r, x, LC);

        mont_limbs(res, r, m, m_inv, LC);
    }

    template <class storage>
    static inline void _rshift(storage &res, const storage &value, unsigned LC, unsigned LB)
    {
        limb_t LB_minus_1 = LB - 1;

        for (uint i = 0; i < LC - 1; i++)
        {
            res.limbs[i] = (value.limbs[i + 1] << LB_minus_1) | (value.limbs[i] >> 1);
        }
        res.limbs[LC - 1] = value.limbs[LC - 1] >> 1;
    }

    template <class storage>
    static inline void div_by_2_mod(storage &res, const storage &value, unsigned LC, unsigned LB)
    {
        _rshift(res, value, LC, LB);
    }

    template <class storage>
    static inline storage inverse(storage &u, storage &v, storage &b, storage &c, const storage &m, limb_t m_inv, const storage one, unsigned LC, unsigned LB)
    {
        while (!(is_equal(u, one, LC)) && !(is_equal(v, one, LC)))
        {
            while ((u.limbs[0] & 0x1) == 0)
            {
                div_by_2_mod(u, u, LC, LB);

                if ((b.limbs[0] & 0x1) != 0)
                {
                    add_limbs_host(b, b, m, LC);
                }

                div_by_2_mod(b, b, LC, LB);
            }

            while ((v.limbs[0] & 0x1) == 0)
            {
                div_by_2_mod(v, v, LC, LB);

                if ((c.limbs[0] & 0x1) != 0)
                {
                    add_limbs_host(c, c, m, LC);
                }
                div_by_2_mod(c, c, LC, LB);
            }

            if (is_gt_limbs(u, v, LC))
            {
                sub_limbs_host(u, u, v, LC);

                sub_mod_limbs_host(b, b, c, m, LC);
            }
            else
            {
                sub_limbs_host(v, v, u, LC);

                sub_mod_limbs_host(c, c, b, m, LC);
            }
        }

        if (is_equal(u, one, LC))
        {
            return b;
        }
        else
        {
            return c;
        }
    }
}