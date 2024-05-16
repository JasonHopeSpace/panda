#pragma once

#include <stdio.h>

#include "field/field.cuh"
#include "curve/projective.cuh"

template <typename Field, unsigned B_VALUE> class Projective;

template <typename Field, unsigned B_VALUE>
class Affine
{
    typedef Field field;
    typedef typename Field::storage storage;
    typedef typename Field::storage_wide storage_wide;

    public:
        field x;
        field y;

    static HOST_DEVICE_INLINE Affine to_montgomery(const Affine &point)
    {
        field x = field::to_montgomery(point.x);
        field y = field::to_montgomery(point.y);

        return {x, y};
    }  

    static HOST_DEVICE_INLINE Affine from_montgomery(const Affine &point)
    {
        field x = field::from_montgomery(point.x);
        field y = field::from_montgomery(point.y);

        return {x, y};
    }

    static HOST_DEVICE_INLINE Affine neg(const Affine &point)
    {
        field y = field::neg(point.y);
    
        return {point.x, y};
    }

    static HOST_DEVICE_INLINE Projective<Field, B_VALUE> to_projective(const Affine &point)
    {
        field z = field::get_field_one();

        return Projective<Field, B_VALUE>{point.x, point.y, z};
    }

    //  y^2=x^3+b
    #if 0
    static HOST_DEVICE_INLINE bool is_on_curve(const Affine &point)
    {
        const field x = point.x;
        const field y = point.y;

        const field y2 = y^2;
        const field x2 = x^2;
        const field x3 = x * x2;
        const field a = y2;

        field bone;
        bone.limbs_storage = field::mul_storage(B_VALUE, fd.get_one());
        
        const field b = x3 + bone;

        return a == b;
    }
    #endif

    static /* HOST_DEVICE_INLINE */ __host__ __device__  bool is_zero(const Affine &point)
    {
        return field::is_zero(point.x);
    }

    static constexpr HOST_DEVICE_INLINE void print(const Affine &point)
    {
        printf("  Affine {\n");
        printf("    x:");
        Field::print(point.x);
        printf("    y:");
        Field::print(point.y);
        printf("  }\n");
    }

    static constexpr HOST_DEVICE_INLINE void print(const char *info, const Affine &point)
    {
        printf("  %s\n", info);
        printf("  Affine {\n");
        printf("    x:");
        Field::print(point.x);
        printf("    y:");
        Field::print(point.y);
        printf("  }\n");
    }
};