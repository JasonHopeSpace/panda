extern crate core;

use ark_bn254::{Fq, G1Affine, G1Projective};
use ark_ec::{msm::VariableBaseMSM, AffineCurve, ProjectiveCurve};
use ark_ff::{BigInteger, BigInteger256, PrimeField};
use ark_std::{end_timer, start_timer, UniformRand};
use panda::{
    gpu_manager::{
        unit::{panda_msm_bn254_gpu, panda_msm_bn254_gpu_host},
        {PandaGpuManager, PandaGpuManagerInitUnitType},
    },
    utils::{transmute_to_value, transmute_values},
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::usize;

pub fn generate_points_scalars<G: AffineCurve>(len: usize) -> (Vec<G>, Vec<G::ScalarField>) {
    // let points = <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(
    //     &(0..len)
    //         .map(|_| G::Projective::prime_subgroup_generator())
    //         .collect::<Vec<_>>(),
    // );

    let mut rng = ChaCha20Rng::from_entropy();

    let points = <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(
        &(0..len)
            .map(|_| G::Projective::rand(&mut rng))
            .collect::<Vec<_>>(),
    );

    // let scalars = (0..len).map(|_| G::ScalarField::one()).collect::<Vec<_>>();

    // let scalars = (0..len)
    // .scan(G::ScalarField::zero(), |acc, _| {
    //     *acc = *acc + G::ScalarField::one();
    //     Some(*acc)
    // })
    // .collect::<Vec<_>>();

    let scalars = (0..len)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();

    (points, scalars)
}

#[test]
fn test_msm_bn254_correctness_device() {
    let min_k = 10;
    let max_k = 20;

    let gm = PandaGpuManager::new(0).unwrap();

    for k in min_k..=max_k {
        let n = 1 << k;

        let (points, scalars) = generate_points_scalars::<G1Affine>(n);

        // CPU
        let timer: ark_std::perf_trace::TimerInfo = start_timer!(|| format!("cpu msm k={}", k));
        let scalar_cpu = scalars.iter().map(|s| s.into_repr()).collect::<Vec<_>>();
        let affine_cpu = VariableBaseMSM::multi_scalar_mul(&points, &scalar_cpu).into_affine();
        end_timer!(timer);
        // println!("cpu result affine.x: {:?}", affine_cpu.x.0.to_bytes_le());
        // println!("cpu result affine.y: {:?}", affine_cpu.y.0.to_bytes_le());

        let points_gpu = points.clone();
        let mut points_gpu_bytes = vec![];

        for p in points_gpu {
            let mut x_bytes = transmute_values(p.x.0.as_ref()).to_vec();
            let mut y_bytes = transmute_values(p.y.0.as_ref()).to_vec();
            points_gpu_bytes.append(&mut x_bytes);
            points_gpu_bytes.append(&mut y_bytes);
        }

        // GPU
        let timer: ark_std::perf_trace::TimerInfo = start_timer!(|| format!("gpu msm k={}", k));
        let scalar_gpu = transmute_values(&scalars);
        let result_bytes_gpu =
            panda_msm_bn254_gpu(&gm, scalar_gpu, points_gpu_bytes.as_slice()).unwrap();
        end_timer!(timer);

        // transmute bytes into G1Projective
        let bytes_len = result_bytes_gpu.len();
        let x =
            transmute_to_value::<BigInteger256>(&mut result_bytes_gpu[..bytes_len / 3].to_vec());
        let y = transmute_to_value::<BigInteger256>(
            &mut result_bytes_gpu[bytes_len / 3..bytes_len * 2 / 3].to_vec(),
        );
        let z = transmute_to_value::<BigInteger256>(
            &mut result_bytes_gpu[bytes_len * 2 / 3..].to_vec(),
        );

        let x = Fq::new(x);
        let y = Fq::new(y);
        let z = Fq::new(z);

        let actual = G1Projective::new(x, y, z);
        let affine_gpu = actual.into_affine();
        // println!("gpu result affine.x: {:?}", affine_gpu.x.0.to_bytes_le());
        // println!("gpu result affine.y: {:?}", affine_gpu.y.0.to_bytes_le());

        assert_eq!(affine_cpu.x, affine_gpu.x);
        assert_eq!(affine_cpu.y, affine_gpu.y);
        assert_eq!(affine_cpu.infinity, affine_gpu.infinity);

        println!("\nRun k = {}, compare successfully\n\n", k);
    }
}

#[test]
fn test_msm_bn254_correctness_host() {
    let min_k = 10;
    let max_k = 16;

    for k in min_k..=max_k {
        let n = 1 << k;
        let (points, scalars) = generate_points_scalars::<G1Affine>(n);

        let points_gpu = points.clone();
        let mut points_gpu_bytes = vec![];

        for p in points_gpu {
            let mut x_bytes = transmute_values(p.x.0.as_ref()).to_vec();
            let mut y_bytes = transmute_values(p.y.0.as_ref()).to_vec();
            points_gpu_bytes.append(&mut x_bytes);
            points_gpu_bytes.append(&mut y_bytes);
        }

        let device_manager = PandaGpuManager::init_all(
            0,
            PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeMSM,
            Some(&[points_gpu_bytes.as_slice()]),
            None,
        )
        .unwrap();

        let scalar_cpu = scalars.iter().map(|s| s.into_repr()).collect::<Vec<_>>();
        let result_affine_cpu =
            VariableBaseMSM::multi_scalar_mul(&points, &scalar_cpu).into_affine();
        // println!("result_affine_cpu.x: {:?}", result_affine_cpu.x.0.to_bytes_le());
        // println!("result_affine_cpu.y: {:?}", result_affine_cpu.y.0.to_bytes_le());

        let scalar_gpu = transmute_values(&scalars);

        // dump
        let is_dump_file = false;
        if is_dump_file {
            use std::fs;
            fs::write("./scalars.bin", scalar_gpu).unwrap();
            // dump bases
            fs::write("./bases.bin", points_gpu_bytes.clone()).unwrap();
            let result_combined_data: Vec<u8> = {
                let mut combined = result_affine_cpu.x.0.to_bytes_le().to_vec();
                combined.extend_from_slice(&result_affine_cpu.y.0.to_bytes_le());
                combined
            };
            fs::write("./result_affine.bin", &result_combined_data).unwrap();
        }

        let result_bytes_gpu =
            panda_msm_bn254_gpu_host(&device_manager, scalar_gpu, points_gpu_bytes.as_slice())
                .unwrap();

        // transmute bytes into G1Projective
        let bytes_len = result_bytes_gpu.len();
        let x =
            transmute_to_value::<BigInteger256>(&mut result_bytes_gpu[..bytes_len / 3].to_vec());
        let y = transmute_to_value::<BigInteger256>(
            &mut result_bytes_gpu[bytes_len / 3..bytes_len * 2 / 3].to_vec(),
        );
        let z = transmute_to_value::<BigInteger256>(
            &mut result_bytes_gpu[bytes_len * 2 / 3..].to_vec(),
        );

        let x = Fq::new(x);
        let y = Fq::new(y);
        let z = Fq::new(z);

        let actual = G1Projective::new(x, y, z);
        let result_affine_gpu = actual.into_affine();

        // println!("actual_affine.x: {:?}", result_affine_gpu.x.0.to_bytes_le());
        // println!("actual_affine.y: {:?}", result_affine_gpu.y.0.to_bytes_le());
        assert_eq!(result_affine_cpu.x, result_affine_gpu.x);
        assert_eq!(result_affine_cpu.y, result_affine_gpu.y);
        assert_eq!(result_affine_cpu.infinity, result_affine_gpu.infinity);

        println!("\nRun k = {}, compare successfully\n\n", k);
    }
}
