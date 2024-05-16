use panda_halo2_zal::PandaEngine;

use ark_std::{end_timer, start_timer};
use rand_core::OsRng;

use halo2_middleware::halo2curves::bn256;
use halo2_middleware::halo2curves::ff::Field;
use halo2_middleware::halo2curves::group::prime::PrimeCurveAffine;
use halo2_middleware::halo2curves::group::{Curve, Group};
use halo2_middleware::halo2curves::msm::best_multiexp;
use halo2_middleware::halo2curves::zal::MsmAccel;

fn run_msm_zal(min_k: usize, max_k: usize) {
    let points = (0..1 << max_k)
        .map(|_| bn256::G1::random(OsRng))
        .collect::<Vec<_>>();
    let mut affine_points = vec![bn256::G1Affine::identity(); 1 << max_k];
    bn256::G1::batch_normalize(&points[..], &mut affine_points[..]);
    let points = affine_points;

    let scalars = (0..1 << max_k)
        .map(|_| bn256::Fr::random(OsRng))
        .collect::<Vec<_>>();

    for k in min_k..=max_k {
        let points = &points[..1 << k];
        let scalars = &scalars[..1 << k];

        let t0 = start_timer!(|| format!("freestanding msm k={}", k));
        let e0 = best_multiexp(scalars, points);
        end_timer!(t0);

        let engine = PandaEngine::new();
        let t1 = start_timer!(|| format!("PandaEngine msm k={}", k));
        let e1 = engine.msm(scalars, points);
        end_timer!(t1);

        assert_eq!(e0, e1);

        // Caching API
        // -----------
        let t2 = start_timer!(|| format!("PandaEngine msm cached base k={}", k));
        let base_descriptor = engine.get_base_descriptor(points);
        let e2 = engine.msm_with_cached_base(scalars, &base_descriptor);
        end_timer!(t2);

        assert_eq!(e0, e2)
    }
}

#[test]
fn t_msm_zal() {
    run_msm_zal(3, 14);
}
