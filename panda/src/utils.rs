pub fn transmute_values<'a, U: std::fmt::Debug>(values: &'a [U]) -> &'a [u8] {
    let ptr = values.as_ptr();
    let len = values.len();

    assert!(
        (ptr as usize) % std::mem::align_of::<u8>() == 0,
        "trying to cast with mismatched layout"
    );

    let size = std::mem::size_of::<U>() * len;
    let out: &'a [u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };

    out
}

pub fn transmute_mut_values<'a, U: std::fmt::Debug>(values: &'a mut [U]) -> &'a [u8] {
    let ptr = values.as_ptr();
    let len = values.len();

    assert!(
        (ptr as usize) % std::mem::align_of::<u8>() == 0,
        "trying to cast with mismatched layout"
    );

    let size = std::mem::size_of::<U>() * len;
    let mut out: &'a [u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };

    out
}

pub fn transmute_to_value<'a, U: std::fmt::Debug + Clone>(bytes_data: &mut Vec<u8>) -> U {
    let size = std::mem::size_of::<u8>() * bytes_data.len();
    let bytes_data_ptr = bytes_data.as_mut_ptr();

    let mut curve_value = Vec::<U>::with_capacity(1);
    let curve_value_ptr = curve_value.as_mut_ptr() as *mut u8;

    unsafe {
        std::ptr::copy_nonoverlapping(bytes_data_ptr, curve_value_ptr, size);
    }
    std::mem::forget(bytes_data_ptr);
    unsafe { curve_value.set_len(1) };

    curve_value[0].clone()
}

pub fn transmute_to_values<'a, U: std::fmt::Debug + Clone>(bytes_data: &mut Vec<u8>) -> Vec<U> {
    let size = bytes_data.len();

    let bytes_data_ptr = bytes_data.as_mut_ptr();

    let mut curve_value = Vec::<U>::with_capacity(size / 32);
    let curve_value_ptr = curve_value.as_mut_ptr() as *mut u8;
    unsafe {
        std::ptr::copy(bytes_data_ptr, curve_value_ptr, size);
    }
    std::mem::forget(bytes_data_ptr);
    unsafe { curve_value.set_len(size / 32) };

    curve_value
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_ec::{AffineCurve, ProjectiveCurve};
    use ark_ff::{BigInteger, BigInteger256, Field, One, PrimeField};

    #[test]
    fn test_transmute() {
        use ark_bls12_377::Fr;

        let n: usize = 1 << 6;
        let scalars = (0..n).into_iter().map(|_| Fr::one()).collect::<Vec<_>>();

        let mut scalar_bytes = transmute_values(&scalars).to_vec();

        let actual = transmute_to_values::<Fr>(&mut scalar_bytes);
        let expect = scalars;
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_transmute_curve_affine() {
        use ark_bn254::{Fq, G1Affine, G1Projective};

        let point = G1Projective::prime_subgroup_generator();
        println!("raw.x: {:?}", point.x.to_string());
        println!("raw.y: {:?}", point.y.to_string());
        println!("raw.y: {:?}", point.z.to_string());

        let expect = point.into_affine();

        println!("\n===Field_to_Bytes");
        let mut x_bytes = transmute_values(point.x.0.as_ref()).to_vec();
        let mut y_bytes = transmute_values(&[point.y]).to_vec();
        let mut z_bytes = transmute_values(&[point.z]).to_vec();
        println!("point.x: {:?}", point.x.0.to_bytes_le());
        println!("point.y: {:?}", point.y.0.to_bytes_le());
        println!("x_bytes: {:?}", x_bytes);
        println!("y_bytes: {:?}", y_bytes);
        x_bytes.append(&mut y_bytes);
        x_bytes.append(&mut z_bytes);

        let mut points_bytes = x_bytes;

        println!("\n===Bytes_to_Field");
        let points_bytes_len = points_bytes.len();
        let x =
            transmute_to_value::<BigInteger256>(&mut points_bytes[..points_bytes_len / 3].to_vec());
        let y = transmute_to_value::<BigInteger256>(
            &mut points_bytes[points_bytes_len / 3..points_bytes_len * 2 / 3].to_vec(),
        );

        let z = transmute_to_value::<BigInteger256>(
            &mut points_bytes[points_bytes_len * 2 / 3..].to_vec(),
        );
        println!("BigInteger256.new.x: {:?}", x.to_bytes_le());

        let x = Fq::new(x);
        let y = Fq::new(y);
        let z = Fq::new(z);
        println!("new.x: {:?}", x.0.to_bytes_le());
        println!("new.y: {:?}", y.0.to_bytes_le());

        println!("\n===Assert");
        let p2 = G1Projective::new(x, y, z);
        let actual = p2.into_affine();
        assert_eq!(expect, actual);
        println!("Success");
    }
}
