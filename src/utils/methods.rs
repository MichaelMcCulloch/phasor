use candle_core::{DType, Device, Result, Shape, Tensor, WithDType};
use core::f64;

#[inline]
pub fn generic_add<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.add(rhs)
}

#[inline]
pub fn generic_sub<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.sub(rhs)
}

#[inline]
pub fn generic_mul<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.mul(rhs)
}

#[inline]
pub fn generic_div<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.div(rhs)
}

#[inline]
pub fn generic_clamp<T: WithDType>(tensor: &Tensor, min: &T, max: &T) -> Result<Tensor> {
    tensor.clamp(*min, *max)
}

#[inline]
pub fn generic_add_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_add(&scalar_tensor)
}

#[inline]
pub fn generic_sub_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_sub(&scalar_tensor)
}

#[inline]
pub fn generic_mul_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_mul(&scalar_tensor)
}

#[inline]
pub fn generic_div_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_div(&scalar_tensor)
}

#[inline]
pub fn generic_pow_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_pow(&scalar_tensor)
}

#[inline]
pub fn generic_powf<T: WithDType>(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    tensor.powf(exponent)
}

#[inline]
pub fn generic_pow<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.pow(rhs)
}

#[inline]
pub fn generic_sin<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.sin()
}

#[inline]
pub fn generic_cos<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.cos()
}

#[inline]
pub fn generic_sinh<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    let exp_tensor = generic_exp::<T>(tensor)?;
    let neg_exp_tensor = generic_exp::<T>(&generic_neg::<T>(tensor)?)?;
    generic_div_scalar::<T>(
        &generic_sub::<T>(&exp_tensor, &neg_exp_tensor)?,
        T::from_f64(2.0),
    )
}

#[inline]
pub fn generic_cosh<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    let exp_tensor = generic_exp::<T>(tensor)?;
    let neg_exp_tensor = generic_exp::<T>(&generic_neg::<T>(tensor)?)?;
    generic_div_scalar::<T>(
        &generic_add::<T>(&exp_tensor, &neg_exp_tensor)?,
        T::from_f64(2.0),
    )
}

#[inline]
pub fn generic_tanh<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.tanh()
}

#[inline]
fn generic_arctan_approx<T: WithDType>(x: &Tensor) -> Result<Tensor> {
    let x2 = generic_mul::<T>(x, x)?;
    let x3 = generic_mul::<T>(&x2, x)?;
    let x5 = generic_mul::<T>(&x3, &x2)?;
    let x7 = generic_mul::<T>(&x5, &x2)?;
    let x9 = generic_mul::<T>(&x7, &x2)?;
    let x11 = generic_mul::<T>(&x9, &x2)?;
    let term1 = x.clone();
    let term2 = generic_div_scalar::<T>(&x3, T::from_f64(3.0))?;
    let term3 = generic_div_scalar::<T>(&x5, T::from_f64(5.0))?;
    let term4 = generic_div_scalar::<T>(&x7, T::from_f64(7.0))?;
    let term5 = generic_div_scalar::<T>(&x9, T::from_f64(9.0))?;
    let term6 = generic_div_scalar::<T>(&x11, T::from_f64(11.0))?;
    generic_add::<T>(
        &generic_sub::<T>(
            &generic_add::<T>(&term1, &term3)?,
            &generic_add::<T>(&term2, &term4)?,
        )?,
        &generic_sub::<T>(&term5, &term6)?,
    )
}

#[inline]
pub fn generic_atan<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    let one = generic_ones::<T>(tensor.device(), T::DTYPE, tensor.shape())?;
    generic_atan2::<T>(tensor, &one)
}

#[inline]
pub fn generic_atan2<T: WithDType>(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    let zero = generic_zeros::<T>(y.device(), T::DTYPE, y.shape())?;
    let eps = generic_mul_scalar::<T>(
        &generic_ones::<T>(y.device(), T::DTYPE, y.shape())?,
        T::from_f64(1e-15),
    )?;
    let pi = generic_mul_scalar::<T>(
        &generic_ones::<T>(y.device(), T::DTYPE, y.shape())?,
        T::from_f64(std::f64::consts::PI),
    )?;
    let pi_half = generic_mul_scalar::<T>(&pi, T::from_f64(0.5))?;

    let y_mag = generic_abs::<T>(y)?;
    let x_mag = generic_abs::<T>(x)?;

    let both_zero = generic_mul::<T>(
        &generic_lt::<T>(&y_mag, &eps)?,
        &generic_lt::<T>(&x_mag, &eps)?,
    )?;
    let x_zero = generic_lt::<T>(&x_mag, &eps)?;

    let base_atan = generic_where::<T>(
        &x_zero,
        &zero,
        &generic_arctan_approx::<T>(&generic_div::<T>(y, x)?)?,
    )?;

    let x_lt_zero = generic_lt::<T>(x, &zero)?;
    let y_lt_zero = generic_lt::<T>(y, &zero)?;
    let adjustment = generic_where::<T>(
        &x_lt_zero,
        &generic_where::<T>(&y_lt_zero, &generic_neg::<T>(&pi)?, &pi)?,
        &zero,
    )?;
    let adjusted_atan = generic_add::<T>(&base_atan, &adjustment)?;

    let x_zero_result = generic_where::<T>(&y_lt_zero, &generic_neg::<T>(&pi_half)?, &pi_half)?;

    generic_where::<T>(
        &both_zero,
        &zero,
        &generic_where::<T>(&x_zero, &x_zero_result, &adjusted_atan)?,
    )
}

#[inline]
pub fn generic_neg<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.neg()
}

#[inline]
pub fn generic_abs<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.abs()
}
#[inline]
pub fn generic_max<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.maximum(rhs)
}
#[inline]
pub fn generic_exp<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.exp()
}

#[inline]
pub fn generic_log<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.log()
}

#[inline]
pub fn generic_mean<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.mean_all()
}

#[inline]
pub fn generic_lt<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.lt(rhs)
}

#[inline]
pub fn generic_lte<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.le(rhs)
}

#[inline]
pub fn generic_eq<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.eq(rhs)
}

#[inline]
pub fn generic_ne<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(rhs)
}

#[inline]
pub fn generic_gt<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.gt(rhs)
}

#[inline]
pub fn generic_gte<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ge(rhs)
}

#[inline]
pub fn generic_where<T: WithDType>(
    condition: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
) -> Result<Tensor> {
    condition.where_cond(on_true, on_false)
}

#[inline]
pub fn generic_and<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.mul(rhs)
}

#[inline]
pub fn generic_or<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(0u8)?.add(&rhs.ne(0u8)?)?.ne(0u8)
}

#[inline]
pub fn generic_xor<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(rhs)
}

#[inline]
pub fn generic_not<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.eq(0u8)
}

#[inline]
pub fn generic_zeros<T: WithDType>(device: &Device, dtype: DType, shape: &Shape) -> Result<Tensor> {
    Tensor::zeros(shape, dtype, device)
}

#[inline]
pub fn generic_ones<T: WithDType>(device: &Device, dtype: DType, shape: &Shape) -> Result<Tensor> {
    Tensor::ones(shape, dtype, device)
}

#[inline]
pub fn generic_transpose<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.t()
}

#[inline]
pub fn generic_matmul<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.matmul(rhs)
}

#[inline]
pub fn generic_sum<T: WithDType>(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    tensor.sum(dim)
}

#[inline]
pub fn generic_broadcast<T: WithDType>(tensor: &Tensor, shape: &Shape) -> Result<Tensor> {
    tensor.broadcast_as(shape)
}

#[inline]
pub fn generic_complex_add<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((lhs_real.add(rhs_real)?, lhs_imag.add(rhs_imag)?))
}

#[inline]
pub fn generic_complex_sub<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((lhs_real.sub(rhs_real)?, lhs_imag.sub(rhs_imag)?))
}

#[inline]
pub fn generic_complex_mul<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let real = lhs_real.mul(rhs_real)?.sub(&lhs_imag.mul(rhs_imag)?)?;
    let imag = lhs_real.mul(rhs_imag)?.add(&lhs_imag.mul(rhs_real)?)?;
    Ok((real, imag))
}

#[inline]
pub fn generic_complex_div<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let denom = rhs_real.mul(rhs_real)?.add(&rhs_imag.mul(rhs_imag)?)?;
    let real = lhs_real
        .mul(rhs_real)?
        .add(&lhs_imag.mul(rhs_imag)?)?
        .div(&denom)?;
    let imag = lhs_imag
        .mul(rhs_real)?
        .sub(&lhs_real.mul(rhs_imag)?)?
        .div(&denom)?;
    Ok((real, imag))
}

#[inline]
pub fn generic_complex_clamp<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    min: &T,
    max: &T,
) -> Result<(Tensor, Tensor)> {
    Ok((real.clamp(*min, *max)?, imag.clamp(*min, *max)?))
}

#[inline]
pub fn generic_complex_mul_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((
        real.broadcast_mul(&scalar_tensor)?,
        imag.broadcast_mul(&scalar_tensor)?,
    ))
}

#[inline]
pub fn generic_complex_div_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((
        real.broadcast_div(&scalar_tensor)?,
        imag.broadcast_div(&scalar_tensor)?,
    ))
}

#[inline]
pub fn generic_complex_add_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((real.broadcast_add(&scalar_tensor)?, imag.clone()))
}

#[inline]
pub fn generic_complex_sub_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((real.broadcast_sub(&scalar_tensor)?, imag.clone()))
}

#[inline]
pub fn generic_complex_pow_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let r = generic_hypot::<T>(real, imag)?;
    let theta = generic_atan2::<T>(imag, real)?;
    let r_pow = generic_pow_scalar::<T>(&r, scalar)?;
    let new_theta = generic_mul_scalar::<T>(&theta, scalar)?;
    Ok((
        generic_mul::<T>(&r_pow, &generic_cos::<T>(&new_theta)?)?,
        generic_mul::<T>(&r_pow, &generic_sin::<T>(&new_theta)?)?,
    ))
}

#[inline]
pub fn generic_complex_powf<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    exponent: f64,
) -> Result<(Tensor, Tensor)> {
    let r = generic_hypot::<T>(real, imag)?;
    let theta = generic_atan2::<T>(imag, real)?;
    let r_pow = generic_powf::<T>(&r, exponent)?;
    let new_theta = generic_mul_scalar::<T>(&theta, T::from_f64(exponent))?;
    Ok((
        generic_mul::<T>(&r_pow, &generic_cos::<T>(&new_theta)?)?,
        generic_mul::<T>(&r_pow, &generic_sin::<T>(&new_theta)?)?,
    ))
}

#[inline]
pub fn generic_complex_pow<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (log_real, log_imag) = generic_complex_log::<T>(real, imag)?;
    let (mul_real, mul_imag) =
        generic_complex_mul::<T>(&log_real, &log_imag, other_real, other_imag)?;
    generic_complex_exp::<T>(&mul_real, &mul_imag)
}

#[inline]
pub fn generic_complex_sin<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    Ok((
        generic_mul::<T>(&generic_sin::<T>(real)?, &generic_cosh::<T>(imag)?)?,
        generic_mul::<T>(&generic_cos::<T>(real)?, &generic_sinh::<T>(imag)?)?,
    ))
}

#[inline]
pub fn generic_complex_cos<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    Ok((
        generic_mul::<T>(&generic_cos::<T>(real)?, &generic_cosh::<T>(imag)?)?,
        generic_neg::<T>(&generic_mul::<T>(
            &generic_sin::<T>(real)?,
            &generic_sinh::<T>(imag)?,
        )?)?,
    ))
}

#[inline]
pub fn generic_complex_sinh<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((
        generic_mul::<T>(&generic_sinh::<T>(real)?, &generic_cos::<T>(imag)?)?,
        generic_mul::<T>(&generic_cosh::<T>(real)?, &generic_sin::<T>(imag)?)?,
    ))
}

#[inline]
pub fn generic_complex_cosh<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((
        generic_mul::<T>(&generic_cosh::<T>(real)?, &generic_cos::<T>(imag)?)?,
        generic_mul::<T>(&generic_sinh::<T>(real)?, &generic_sin::<T>(imag)?)?,
    ))
}

#[inline]
pub fn generic_complex_tanh<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let two_x = generic_mul_scalar::<T>(real, T::from_f64(2.0))?;
    let two_y = generic_mul_scalar::<T>(imag, T::from_f64(2.0))?;

    let sinh_2x = generic_sinh::<T>(&two_x)?;
    let sin_2y = generic_sin::<T>(&two_y)?;
    let cosh_2x = generic_cosh::<T>(&two_x)?;
    let cos_2y = generic_cos::<T>(&two_y)?;

    let denom = generic_add::<T>(&cosh_2x, &cos_2y)?;

    Ok((
        generic_div::<T>(&sinh_2x, &denom)?,
        generic_div::<T>(&sin_2y, &denom)?,
    ))
}

#[inline]
pub fn generic_complex_atan<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let zero = generic_zeros::<T>(real.device(), T::DTYPE, real.shape())?;
    let one = generic_ones::<T>(real.device(), T::DTYPE, real.shape())?;
    let eps = generic_mul_scalar::<T>(
        &generic_ones::<T>(real.device(), T::DTYPE, real.shape())?,
        T::from_f64(1e-15),
    )?;
    let pi_half = generic_mul_scalar::<T>(
        &generic_ones::<T>(real.device(), T::DTYPE, real.shape())?,
        T::from_f64(std::f64::consts::PI / 2.0),
    )?;

    let iz_real = generic_neg::<T>(imag)?;
    let iz_imag = real.clone();

    let numerator_real = one.clone();
    let numerator_imag = zero.clone();
    let (numerator_real, numerator_imag) =
        generic_complex_add::<T>(&numerator_real, &numerator_imag, &iz_real, &iz_imag)?;

    let denominator_real = one.clone();
    let denominator_imag = zero.clone();
    let (denominator_real, denominator_imag) =
        generic_complex_sub::<T>(&denominator_real, &denominator_imag, &iz_real, &iz_imag)?;

    let mag_real = generic_abs::<T>(real)?;
    let mag_imag = generic_abs::<T>(imag)?;

    let is_zero = generic_mul::<T>(
        &generic_lt::<T>(&mag_real, &eps)?,
        &generic_lt::<T>(&mag_imag, &eps)?,
    )?;

    let near_i = generic_mul::<T>(
        &generic_lt::<T>(&mag_real, &eps)?,
        &generic_lt::<T>(
            &generic_abs::<T>(&generic_sub_scalar::<T>(&mag_imag, T::one())?)?,
            &eps,
        )?,
    )?;

    let (ratio_real, ratio_imag) = generic_complex_div::<T>(
        &numerator_real,
        &numerator_imag,
        &denominator_real,
        &denominator_imag,
    )?;
    let (log_real, log_imag) = generic_complex_log::<T>(&ratio_real, &ratio_imag)?;
    let standard_real = generic_mul_scalar::<T>(&log_imag, T::from_f64(0.5))?;
    let standard_imag = generic_mul_scalar::<T>(&log_real, T::from_f64(-0.5))?;

    let zero_real = zero.clone();
    let zero_imag = zero.clone();

    let i_real = generic_mul::<T>(&pi_half, &generic_div::<T>(imag, &mag_imag)?)?;
    let i_imag = generic_mul_scalar::<T>(
        &generic_ones::<T>(real.device(), T::DTYPE, real.shape())?,
        T::from_f64(f64::INFINITY),
    )?;

    let (real, imag) =
        generic_where_complex::<T>(&near_i, &i_real, &i_imag, &standard_real, &standard_imag)?;
    let (result_real, result_imag) =
        generic_where_complex::<T>(&is_zero, &zero_real, &zero_imag, &real, &imag)?;

    Ok((result_real, result_imag))
}

#[inline]
pub fn generic_complex_atan2<T: WithDType>(
    y_real: &Tensor,
    y_imag: &Tensor,
    x_real: &Tensor,
    x_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let arg = generic_atan2::<T>(y_imag, y_real)?;
    let zero = generic_zeros::<T>(y_real.device(), T::DTYPE, y_real.shape())?;
    Ok((arg, zero))
}

#[inline]
pub fn generic_complex_exp<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    let exp_real = generic_exp::<T>(real)?;
    Ok((
        generic_mul::<T>(&exp_real, &generic_cos::<T>(imag)?)?,
        generic_mul::<T>(&exp_real, &generic_sin::<T>(imag)?)?,
    ))
}

#[inline]
pub fn generic_complex_log<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    let abs = generic_hypot::<T>(real, imag)?;
    let arg = generic_atan2::<T>(imag, real)?;
    Ok((generic_log::<T>(&abs)?, arg))
}

#[inline]
pub fn generic_hypot<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<Tensor> {
    let real_sq = generic_mul::<T>(real, real)?;
    let imag_sq = generic_mul::<T>(imag, imag)?;
    let sum_sq = generic_add::<T>(&real_sq, &imag_sq)?;
    let epsilon = T::from_f64(1e-15);
    let sum_sq_plus_epsilon = generic_add_scalar::<T>(&sum_sq, epsilon)?;
    generic_powf::<T>(&sum_sq_plus_epsilon, 0.5)
}

#[inline]
pub fn generic_where_complex<T: WithDType>(
    condition: &Tensor,
    on_true_real: &Tensor,
    on_true_imag: &Tensor,
    on_false_real: &Tensor,
    on_false_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((
        condition.where_cond(on_true_real, on_false_real)?,
        condition.where_cond(on_true_imag, on_false_imag)?,
    ))
}

#[inline]
pub fn generic_complex_lt<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_lt::<T>(&mag_self, &mag_other)
}

#[inline]
pub fn generic_complex_lte<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_lte::<T>(&mag_self, &mag_other)
}

#[inline]
pub fn generic_complex_eq<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let real_eq = generic_eq::<T>(real, other_real)?;
    let imag_eq = generic_eq::<T>(imag, other_imag)?;
    generic_mul::<T>(&real_eq, &imag_eq)
}

#[inline]
pub fn generic_complex_ne<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let real_ne = generic_ne::<T>(real, other_real)?;
    let imag_ne = generic_ne::<T>(imag, other_imag)?;
    generic_gt::<T>(
        &generic_add::<T>(&real_ne, &imag_ne)?,
        &generic_zeros::<T>(real.device(), T::DTYPE, real.shape())?,
    )
}

#[inline]
pub fn generic_complex_gt<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_gt::<T>(&mag_self, &mag_other)
}

#[inline]
pub fn generic_complex_gte<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_gte::<T>(&mag_self, &mag_other)
}

#[inline]
pub fn generic_complex_and<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.mul(rhs)
}

#[inline]
pub fn generic_complex_or<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(0u8)?.add(&rhs.ne(0u8)?)?.ne(0u8)
}

#[inline]
pub fn generic_complex_xor<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(rhs)
}

#[inline]
pub fn generic_complex_not<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.eq(0u8)
}

#[cfg(test)]
mod test {
    use crate::utils::methods::*;
    use approx::{assert_relative_eq, RelativeEq};
    use candle_core::{DType, Device, Result, Shape, Tensor};
    use core::f64;
    use std::f64::consts::PI;
    use std::fmt::Debug;
    pub fn assert_relative_eq_vec<T: RelativeEq + Debug>(lhs: Vec<T>, rhs: Vec<T>) {
        lhs.iter()
            .zip(rhs)
            .for_each(|(l, r)| assert_relative_eq!(l, &r));
    }
    pub fn assert_relative_eq_vec_vec<T: RelativeEq + Debug>(lhs: Vec<Vec<T>>, rhs: Vec<Vec<T>>) {
        lhs.iter()
            .flatten()
            .zip(rhs.iter().flatten())
            .for_each(|(l, r)| assert_relative_eq!(l, &r));
    }
    fn create_tensor<T: WithDType>(data: Vec<T>, shape: Shape, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(data, shape, device)
    }

    #[test]
    fn test_generic_add() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let result = generic_add::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_sub() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_sub::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![3.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_mul() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![4.0, 5.0], Shape::from((1, 2)), &device)?;
        let result = generic_mul::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![8.0, 15.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_div() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![10.0, 12.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_div::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![5.0, 4.0]]);
        Ok(())
    }
    #[test]
    fn test_generic_complex_exp() -> Result<()> {
        let device = Device::Cpu;

        // Test case 1: exp(0 + 0i) = 1 + 0i
        let real = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_exp::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 1.0);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 0.0);

        // Test case 2: exp(1 + 0i) = e + 0i
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_exp::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], std::f64::consts::E);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 0.0);

        // Test case 3: exp(0 + pi/2 i) = 0 + 1i
        let real = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(
            vec![std::f64::consts::PI / 2.0],
            Shape::from((1, 1)),
            &device,
        )?;
        let (real_result, imag_result) = generic_complex_exp::<f64>(&real, &imag)?;
        assert_relative_eq!(
            real_result.to_vec2::<f64>()?[0][0],
            0.0,
            max_relative = 1e-6
        );
        assert_relative_eq!(
            imag_result.to_vec2::<f64>()?[0][0],
            1.0,
            max_relative = 1e-6
        );

        // Test case 4: exp(1 + pi/4 i)
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(
            vec![std::f64::consts::PI / 4.0],
            Shape::from((1, 1)),
            &device,
        )?;
        let (real_result, imag_result) = generic_complex_exp::<f64>(&real, &imag)?;
        let expected_real = std::f64::consts::E * (std::f64::consts::PI / 4.0).cos();
        let expected_imag = std::f64::consts::E * (std::f64::consts::PI / 4.0).sin();
        assert_relative_eq!(
            real_result.to_vec2::<f64>()?[0][0],
            expected_real,
            max_relative = 1e-6
        );
        assert_relative_eq!(
            imag_result.to_vec2::<f64>()?[0][0],
            expected_imag,
            max_relative = 1e-6
        );

        // Test case 5: exp(-1 + pi i)
        let real = create_tensor(vec![-1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![std::f64::consts::PI], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_exp::<f64>(&real, &imag)?;
        assert_relative_eq!(
            real_result.to_vec2::<f64>()?[0][0],
            -std::f64::consts::E.powf(-1.0),
            max_relative = 1e-6
        );
        assert_relative_eq!(
            imag_result.to_vec2::<f64>()?[0][0],
            0.0,
            max_relative = 1e-6
        );

        Ok(())
    }

    #[test]
    fn test_generic_complex_log() -> Result<()> {
        let device = Device::Cpu;

        // Test case 1: log(1 + 0i) = 0 + 0i
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_log::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0,);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], -f64::consts::PI);

        // Test case 2: log(e + 0i) = 1 + 0i
        let real = create_tensor(vec![std::f64::consts::E], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_log::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 1.0);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], -f64::consts::PI);

        // Test case 3: log(0 + 1i) = 0 + pi/2 i
        let real = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_log::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0,);
        assert_relative_eq!(
            imag_result.to_vec2::<f64>()?[0][0],
            std::f64::consts::PI / 2.0,
        );

        // Test case 4: log(1 + 1i) = ln(sqrt(2)) + pi/4 i
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_log::<f64>(&real, &imag)?;
        assert_relative_eq!(
            real_result.to_vec2::<f64>()?[0][0],
            0.5 * std::f64::consts::LN_2,
        );
        assert_relative_eq!(
            imag_result.to_vec2::<f64>()?[0][0],
            std::f64::consts::PI / 4.0,
        );

        // Test case 5: log(-1 + 0i) = ln(1) + pi i
        let real = create_tensor(vec![-1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_log::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0,);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], std::f64::consts::PI,);

        Ok(())
    }
    #[test]
    fn test_generic_clamp() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(
            vec![1.0, 5.0, 2.0, 8.0],
            Shape::from(Shape::from((1, 4))),
            &device,
        )?;
        let min = 3.0;
        let max = 6.0;
        let result = generic_clamp::<f64>(&a, &min, &max)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![3.0, 5.0, 3.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_add_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_add_scalar::<f64>(&a, 3.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_sub_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let result = generic_sub_scalar::<f64>(&a, 2.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_mul_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_mul_scalar::<f64>(&a, 4.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![8.0, 12.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_div_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![10.0, 12.0], Shape::from((1, 2)), &device)?;
        let result = generic_div_scalar::<f64>(&a, 2.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![5.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_pow_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_pow_scalar::<f64>(&a, 2.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 9.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_powf() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_powf::<f64>(&a, 2.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 9.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_pow() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_pow::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 9.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_sin() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, PI / 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_sin::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 1.0);
        Ok(())
    }

    #[test]
    fn test_generic_cos() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, PI], Shape::from((1, 2)), &device)?;
        let result = generic_cos::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 1.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], -1.0);
        Ok(())
    }

    #[test]
    fn test_generic_sinh() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_sinh::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 1.1752011936438014);
        Ok(())
    }

    #[test]
    fn test_generic_cosh() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_cosh::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 1.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 1.5430806348152437);
        Ok(())
    }

    #[test]
    fn test_generic_tanh() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_tanh::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 0.7615941559557649);
        Ok(())
    }

    #[test]
    fn test_generic_atan() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_atan::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], PI / 4.0);
        Ok(())
    }

    #[test]
    fn test_generic_atan2() -> Result<()> {
        let device = Device::Cpu;
        let y = create_tensor(vec![1.0, 1.0, -1.0, -1.0], Shape::from((1, 4)), &device)?;
        let x = create_tensor(vec![1.0, -1.0, 1.0, -1.0], Shape::from((1, 4)), &device)?;
        let result = generic_atan2::<f64>(&y, &x)?;

        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], PI / 4.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 3.0 * PI / 4.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][2], -PI / 4.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][3], -3.0 * PI / 4.0);
        Ok(())
    }

    #[test]
    fn test_generic_neg() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, -2.0], Shape::from((1, 2)), &device)?;
        let result = generic_neg::<f64>(&a)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![-1.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_abs() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![-1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_abs::<f64>(&a)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![1.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_exp() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_exp::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 1.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 2.718281828459045);
        Ok(())
    }

    #[test]
    fn test_generic_log() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_log::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 0.6931471805599453);
        Ok(())
    }

    #[test]
    fn test_generic_mean() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0], Shape::from((1, 3)), &device)?;
        let result = generic_mean::<f64>(&a)?;
        assert_relative_eq!(result.to_scalar::<f64>()?, 2.0);
        Ok(())
    }

    #[test]
    fn test_generic_lt() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_lt::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_lte() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_lte::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_eq() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_eq::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_ne() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_ne::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![0, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_gt() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 1.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_gt::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_gte() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_gte::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_where() -> Result<()> {
        let device = Device::Cpu;
        let condition = create_tensor(vec![1u8, 0u8], Shape::from((1, 2)), &device)?;
        let on_true = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let on_false = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let result = generic_where::<f64>(&condition, &on_true, &on_false)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![1.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_and() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 1u8, 0u8, 0u8], Shape::from((1, 4)), &device)?;
        let b = create_tensor(vec![1u8, 0u8, 1u8, 0u8], Shape::from((1, 4)), &device)?;
        let result = generic_and::<u8>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0, 0, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_or() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 1u8, 0u8, 0u8], Shape::from((1, 4)), &device)?;
        let b = create_tensor(vec![1u8, 0u8, 1u8, 0u8], Shape::from((1, 4)), &device)?;
        let result = generic_or::<u8>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 1, 1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_xor() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 1u8, 0u8, 0u8], Shape::from((1, 4)), &device)?;
        let b = create_tensor(vec![1u8, 0u8, 1u8, 0u8], Shape::from((1, 4)), &device)?;
        let result = generic_xor::<u8>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![0, 1, 1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_not() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 0u8], Shape::from((1, 2)), &device)?;
        let result = generic_not::<u8>(&a)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![0, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_zeros() -> Result<()> {
        let device = Device::Cpu;
        let shape = Shape::from(Shape::from((1, 2)));
        let result = generic_zeros::<f64>(&device, DType::F64, &shape)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![0.0, 0.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_ones() -> Result<()> {
        let device = Device::Cpu;
        let shape = Shape::from(Shape::from((1, 2)));
        let result = generic_ones::<f64>(&device, DType::F64, &shape)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![1.0, 1.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_transpose() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::from((2, 2)), &device)?;
        let result = generic_transpose::<f64>(&a)?;
        assert_eq!(
            result.to_vec2::<f64>()?,
            vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_matmul() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::from((2, 2)), &device)?;
        let b = create_tensor(vec![5.0, 6.0, 7.0, 8.0], Shape::from((2, 2)), &device)?;
        let result = generic_matmul::<f64>(&a, &b)?;
        assert_eq!(
            result.to_vec2::<f64>()?,
            vec![vec![19.0, 22.0], vec![43.0, 50.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_sum() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::from((1, 4)), &device)?;
        let result = generic_sum::<f64>(&a, 1)?;
        assert_eq!(result.to_vec1::<f64>()?, vec![10.0]);
        Ok(())
    }

    #[test]
    fn test_generic_broadcast() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let shape = Shape::from(Shape::from((2, 2)));
        let result = generic_broadcast::<f64>(&a, &shape)?;
        assert_eq!(
            result.to_vec2::<f64>()?,
            vec![vec![1.0, 2.0], vec![1.0, 2.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_complex_add() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_add::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq_vec_vec(real.to_vec2::<f64>()?, vec![vec![6.0, 8.0]]);
        assert_relative_eq_vec_vec(imag.to_vec2::<f64>()?, vec![vec![10.0, 12.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_sub() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_sub::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq_vec_vec(real.to_vec2::<f64>()?, vec![vec![4.0, 4.0]]);
        assert_relative_eq_vec_vec(imag.to_vec2::<f64>()?, vec![vec![4.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_mul() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_mul::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq_vec_vec(real.to_vec2::<f64>()?, vec![vec![-16.0, -20.0]]);
        assert_relative_eq_vec_vec(imag.to_vec2::<f64>()?, vec![vec![22.0, 40.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_div() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_div::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq!(real.to_vec2::<f64>()?[0][0], 0.55);
        assert_relative_eq!(real.to_vec2::<f64>()?[0][1], 0.56);
        assert_relative_eq!(imag.to_vec2::<f64>()?[0][0], 0.05);
        assert_relative_eq!(imag.to_vec2::<f64>()?[0][1], 0.08);
        Ok(())
    }

    #[test]
    fn test_generic_complex_clamp() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0, 5.0, 2.0, 8.0], Shape::from((1, 4)), &device)?;
        let imag = create_tensor(vec![1.0, 5.0, 2.0, 8.0], Shape::from((1, 4)), &device)?;
        let min = 3.0;
        let max = 6.0;
        let (real_result, imag_result) = generic_complex_clamp::<f64>(&real, &imag, &min, &max)?;
        assert_eq!(
            real_result.to_vec2::<f64>()?,
            vec![vec![3.0, 5.0, 3.0, 6.0]]
        );
        assert_eq!(
            imag_result.to_vec2::<f64>()?,
            vec![vec![3.0, 5.0, 3.0, 6.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_complex_mul_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_mul_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![2.0, 4.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![6.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_div_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![2.0, 4.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![6.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_div_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![1.0, 2.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_add_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_add_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_sub_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_sub_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![1.0, 2.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![5.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_pow_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_pow_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 2.0);
        Ok(())
    }

    #[test]
    fn test_generic_complex_powf() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_powf::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 2.0);
        Ok(())
    }

    #[test]
    fn test_generic_complex_pow() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag1 = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let real2 = create_tensor(vec![2.0], Shape::from((1, 1)), &device)?;
        let imag2 = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) =
            generic_complex_pow::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 2.0);
        Ok(())
    }

    #[test]
    fn test_generic_complex_sin() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_sin::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 1.5430806348152437);
        Ok(())
    }

    #[test]
    fn test_generic_complex_cos() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_cos::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.5403023058681398);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 0.0);
        Ok(())
    }

    #[test]
    fn test_generic_complex_sinh() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_sinh::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 0.8414709848078965);
        Ok(())
    }

    #[test]
    fn test_generic_complex_cosh() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![0.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_cosh::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.5403023058681398);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 0.0);
        Ok(())
    }

    #[test]
    fn test_generic_complex_tanh() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_tanh::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.99627207622075);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 0.0);
        Ok(())
    }
}
