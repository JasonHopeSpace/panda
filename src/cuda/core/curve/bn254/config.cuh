#pragma once

#include "paramter.cuh"
#include "curve/projective.cuh"
#include "curve/affine.cuh"

#include "field/field.cuh"

namespace CURVE_BN254
{
    typedef Field<fp_configuration> fp;// Fq
    typedef Field<fr_configuration> fr;// Fr

    typedef Projective<fp, WEIERSTRASS_B> projective;
    typedef Affine<fp, WEIERSTRASS_B> affine;
} // namespace  CURVE_BN254
