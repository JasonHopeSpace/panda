#pragma once


namespace panda_curve
{
    #define SW_AFFINE_BYTES 96

    enum CURVE_TYPE
    {
        CURVE_TYPE_DEFAULT      = 0x0,
        CURVE_TYPE_BLS12_377    = 0x1,
        CURVE_TYPE_BLS12_381    = 0x2,
        CURVE_TYPE_BN256        = 0X3,
    };

    enum CURVE_INPUT_DATA_TYPE
    {
        CURVE_INPUT_DATA_TYPE_DEFAULT   = 0x0,
        CURVE_INPUT_DATA_TYPE_U32       = 0x1,
        CURVE_INPUT_DATA_TYPE_U64       = 0x2,
    };

    enum RESULT_COORDINATE_TYPE
    {
        RESULT_COORDINATE_TYPE_JACOBIAN = 0,
        RESULT_COORDINATE_TYPE_PROJECTIVE,
    };
}