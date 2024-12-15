use candle_core::{DType, Device, Result, Shape, Tensor, WithDType};

// Generic function for element-wise addition
#[inline]
pub fn generic_add<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.add(rhs)
}

// Generic function for element-wise subtraction
#[inline]
pub fn generic_sub<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.sub(rhs)
}

// Generic function for element-wise multiplication
#[inline]
pub fn generic_mul<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.mul(rhs)
}

// Generic function for element-wise division
#[inline]
pub fn generic_div<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.div(rhs)
}

// Generic function for clamping
#[inline]
pub fn generic_clamp<T: WithDType>(tensor: &Tensor, min: &T, max: &T) -> Result<Tensor> {
    tensor.clamp(*min, *max)
}

// Generic function for scalar addition
#[inline]
pub fn generic_add_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), tensor.device())?;
    tensor.broadcast_add(&scalar_tensor)
}

// Generic function for scalar subtraction
#[inline]
pub fn generic_sub_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), tensor.device())?;
    tensor.broadcast_sub(&scalar_tensor)
}

// Generic function for scalar multiplication
#[inline]
pub fn generic_mul_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), tensor.device())?;
    tensor.broadcast_mul(&scalar_tensor)
}

// Generic function for scalar division
#[inline]
pub fn generic_div_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), tensor.device())?;
    tensor.broadcast_div(&scalar_tensor)
}

// Generic function for scalar power
#[inline]
pub fn generic_pow_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), tensor.device())?;
    tensor.broadcast_pow(&scalar_tensor)
}

// Generic function for power with float
#[inline]
pub fn generic_powf<T: WithDType>(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    tensor.powf(exponent)
}

// Generic function for power with another tensor
#[inline]
pub fn generic_pow<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.pow(rhs)
}

// Generic function for sin
#[inline]
pub fn generic_sin<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.sin()
}

// Generic function for cos
#[inline]
pub fn generic_cos<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.cos()
}

// Generic function for sinh
#[inline]
pub fn generic_sinh<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    let exp_tensor = generic_exp::<T>(tensor)?;
    let neg_exp_tensor = generic_exp::<T>(&generic_neg::<T>(tensor)?)?;
    generic_div_scalar::<T>(
        &generic_sub::<T>(&exp_tensor, &neg_exp_tensor)?,
        T::from_f64(2.0),
    )
}

// Generic function for cosh
#[inline]
pub fn generic_cosh<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    let exp_tensor = generic_exp::<T>(tensor)?;
    let neg_exp_tensor = generic_exp::<T>(&generic_neg::<T>(tensor)?)?;
    generic_div_scalar::<T>(
        &generic_add::<T>(&exp_tensor, &neg_exp_tensor)?,
        T::from_f64(2.0),
    )
}

// Generic function for tanh
#[inline]
pub fn generic_tanh<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.tanh()
}

// Generic function for atan
#[inline]
pub fn generic_atan<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    let i = generic_add_scalar::<T>(
        &generic_zeros::<T>(tensor.device(), T::DTYPE, tensor.shape())?,
        T::from_f64(1.0),
    )?;
    let numerator = generic_add::<T>(&i, tensor)?;
    let denominator = generic_sub::<T>(&i, tensor)?;
    generic_mul_scalar::<T>(
        &generic_log::<T>(&generic_div::<T>(&numerator, &denominator)?)?,
        T::from_f64(0.5),
    )
}

// Generic function for atan2

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

    // Compute magnitudes
    let y_mag = generic_abs::<T>(y)?;
    let x_mag = generic_abs::<T>(x)?;

    // Special cases
    let both_zero = generic_mul::<T>(
        &generic_lt::<T>(&y_mag, &eps)?,
        &generic_lt::<T>(&x_mag, &eps)?,
    )?;
    let x_zero = generic_lt::<T>(&x_mag, &eps)?;

    // Base atan computation
    let base_atan = generic_atan::<T>(&generic_div::<T>(y, x)?)?;

    // Quadrant adjustments
    let x_lt_zero = generic_lt::<T>(x, &zero)?;
    let y_gte_zero = generic_gte::<T>(y, &zero)?;
    let adjustment = generic_where::<T>(
        &generic_and::<T>(&x_lt_zero, &y_gte_zero)?,
        &pi,
        &generic_neg::<T>(&pi)?,
    )?;
    let adjusted_atan = generic_add::<T>(&base_atan, &adjustment)?;

    // Handle x = 0 cases
    let x_zero_result = generic_where::<T>(&y_gte_zero, &pi_half, &generic_neg::<T>(&pi_half)?)?;

    // Combine all cases
    generic_where::<T>(
        &both_zero,
        &zero,
        &generic_where::<T>(&x_zero, &x_zero_result, &adjusted_atan)?,
    )
}

// Generic function for negation
#[inline]
pub fn generic_neg<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.neg()
}

// Generic function for absolute value
#[inline]
pub fn generic_abs<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.abs()
}

// Generic function for exponential
#[inline]
pub fn generic_exp<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.exp()
}

// Generic function for logarithm
#[inline]
pub fn generic_log<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.log()
}

// Generic function for mean
#[inline]
pub fn generic_mean<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.mean_all()
}

// Generic function for less than
#[inline]
pub fn generic_lt<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.lt(rhs)
}

// Generic function for less than or equal to
#[inline]
pub fn generic_lte<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.le(rhs)
}

// Generic function for equal to
#[inline]
pub fn generic_eq<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.eq(rhs)
}

// Generic function for not equal to
#[inline]
pub fn generic_ne<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(rhs)
}

// Generic function for greater than
#[inline]
pub fn generic_gt<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.gt(rhs)
}

// Generic function for greater than or equal to
#[inline]
pub fn generic_gte<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ge(rhs)
}

// Generic function for where condition
#[inline]
pub fn generic_where<T: WithDType>(
    condition: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
) -> Result<Tensor> {
    condition.where_cond(on_true, on_false)
}

// Generic function for logical and
#[inline]
pub fn generic_and<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.mul(rhs)
}

// Generic function for logical or
#[inline]
pub fn generic_or<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(0u8)?.add(&rhs.ne(0u8)?)?.ne(0u8)
}

// Generic function for logical xor
#[inline]
pub fn generic_xor<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(rhs)
}

// Generic function for logical not
#[inline]
pub fn generic_not<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.eq(0u8)
}

// Generic function for zeros
#[inline]
pub fn generic_zeros<T: WithDType>(device: &Device, dtype: DType, shape: &Shape) -> Result<Tensor> {
    Tensor::zeros(shape, dtype, device)
}

// Generic function for ones
#[inline]
pub fn generic_ones<T: WithDType>(device: &Device, dtype: DType, shape: &Shape) -> Result<Tensor> {
    Tensor::ones(shape, dtype, device)
}

// Generic function for transpose
#[inline]
pub fn generic_transpose<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.t()
}

// Generic function for matmul
#[inline]
pub fn generic_matmul<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.matmul(rhs)
}

// Generic function for sum along a dimension
#[inline]
pub fn generic_sum<T: WithDType>(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    tensor.sum(dim)
}

// Generic function for broadcast
#[inline]
pub fn generic_broadcast<T: WithDType>(tensor: &Tensor, shape: &Shape) -> Result<Tensor> {
    tensor.broadcast_as(shape)
}

// In utils/methods.rs

// Generic function for complex element-wise addition
#[inline]
pub fn generic_complex_add<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((lhs_real.add(rhs_real)?, lhs_imag.add(rhs_imag)?))
}

// Generic function for complex element-wise subtraction
#[inline]
pub fn generic_complex_sub<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((lhs_real.sub(rhs_real)?, lhs_imag.sub(rhs_imag)?))
}

// Generic function for complex element-wise multiplication
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

// Generic function for complex element-wise division
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

// Generic function for complex clamping
#[inline]
pub fn generic_complex_clamp<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    min: &T,
    max: &T,
) -> Result<(Tensor, Tensor)> {
    Ok((real.clamp(*min, *max)?, imag.clamp(*min, *max)?))
}

// Generic function for complex scalar multiplication
#[inline]
pub fn generic_complex_mul_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), real.device())?;
    Ok((
        real.broadcast_mul(&scalar_tensor)?,
        imag.broadcast_mul(&scalar_tensor)?,
    ))
}

// Generic function for complex scalar division
#[inline]
pub fn generic_complex_div_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), real.device())?;
    Ok((
        real.broadcast_div(&scalar_tensor)?,
        imag.broadcast_div(&scalar_tensor)?,
    ))
}

// Generic function for complex scalar addition
#[inline]
pub fn generic_complex_add_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), real.device())?;
    Ok((real.broadcast_add(&scalar_tensor)?, imag.clone()))
}

// Generic function for complex scalar subtraction
#[inline]
pub fn generic_complex_sub_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), real.device())?;
    Ok((real.broadcast_sub(&scalar_tensor)?, imag.clone()))
}

// Generic function for complex scalar power
#[inline]
pub fn generic_complex_pow_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let r = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let theta = generic_atan2::<T>(imag, real)?;
    let r_pow = generic_pow_scalar::<T>(&r, scalar)?;
    let new_theta = generic_mul_scalar::<T>(&theta, scalar)?;
    Ok((
        generic_mul::<T>(&r_pow, &generic_cos::<T>(&new_theta)?)?,
        generic_mul::<T>(&r_pow, &generic_sin::<T>(&new_theta)?)?,
    ))
}

// Generic function for complex power with float
#[inline]
pub fn generic_complex_powf<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    exponent: f64,
) -> Result<(Tensor, Tensor)> {
    let r = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let theta = generic_atan2::<T>(imag, real)?;
    let r_pow = generic_powf::<T>(&r, exponent)?;
    let new_theta = generic_mul_scalar::<T>(&theta, T::from_f64(exponent))?;
    Ok((
        generic_mul::<T>(&r_pow, &generic_cos::<T>(&new_theta)?)?,
        generic_mul::<T>(&r_pow, &generic_sin::<T>(&new_theta)?)?,
    ))
}

// Generic function for complex power with another complex
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

// Generic function for complex sin
#[inline]
pub fn generic_complex_sin<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    Ok((
        generic_mul::<T>(&generic_sin::<T>(real)?, &generic_cosh::<T>(imag)?)?,
        generic_mul::<T>(&generic_cos::<T>(real)?, &generic_sinh::<T>(imag)?)?,
    ))
}

// Generic function for complex cos
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

// Generic function for complex sinh
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

// Generic function for complex cosh
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

// Generic function for complex tanh
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

// Generic function for complex atan
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

    // Form iz
    let iz_real = generic_neg::<T>(imag)?;
    let iz_imag = real.clone();

    // Compute (1 + iz)/(1 - iz)
    let numerator_real = one.clone();
    let numerator_imag = zero.clone();
    let (numerator_real, numerator_imag) =
        generic_complex_add::<T>(&numerator_real, &numerator_imag, &iz_real, &iz_imag)?;

    let denominator_real = one.clone();
    let denominator_imag = zero.clone();
    let (denominator_real, denominator_imag) =
        generic_complex_sub::<T>(&denominator_real, &denominator_imag, &iz_real, &iz_imag)?;

    // Check for special cases
    let mag_real = generic_abs::<T>(real)?;
    let mag_imag = generic_abs::<T>(imag)?;

    // z = 0 case
    let is_zero = generic_mul::<T>(
        &generic_lt::<T>(&mag_real, &eps)?,
        &generic_lt::<T>(&mag_imag, &eps)?,
    )?;

    // z = ±i case (near branch points)
    let near_i = generic_mul::<T>(
        &generic_lt::<T>(&mag_real, &eps)?,
        &generic_lt::<T>(
            &generic_abs::<T>(&generic_sub_scalar::<T>(&mag_imag, T::one())?)?,
            &eps,
        )?,
    )?;

    // Standard computation
    let (ratio_real, ratio_imag) = generic_complex_div::<T>(
        &numerator_real,
        &numerator_imag,
        &denominator_real,
        &denominator_imag,
    )?;
    let (log_real, log_imag) = generic_complex_log::<T>(&ratio_real, &ratio_imag)?;
    let standard_real = generic_mul_scalar::<T>(&log_imag, T::from_f64(0.5))?;
    let standard_imag = generic_mul_scalar::<T>(&log_real, T::from_f64(-0.5))?;

    // Special case results
    let zero_real = zero.clone();
    let zero_imag = zero.clone();

    let i_real = generic_mul::<T>(&pi_half, &generic_div::<T>(imag, &mag_imag)?)?;
    let i_imag = generic_mul_scalar::<T>(
        &generic_ones::<T>(real.device(), T::DTYPE, real.shape())?,
        T::from_f64(f64::INFINITY),
    )?;

    // Combine results using conditional operations
    let (real, imag) =
        generic_where_complex::<T>(&near_i, &i_real, &i_imag, &standard_real, &standard_imag)?;
    let (result_real, result_imag) =
        generic_where_complex::<T>(&is_zero, &zero_real, &zero_imag, &real, &imag)?;

    Ok((result_real, result_imag))
}

// Generic function for complex atan2
#[inline]
pub fn generic_complex_atan2<T: WithDType>(
    y_real: &Tensor,
    y_imag: &Tensor,
    x_real: &Tensor,
    x_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let zero = generic_zeros::<T>(y_real.device(), T::DTYPE, y_real.shape())?;
    let eps = generic_mul_scalar::<T>(
        &generic_ones::<T>(y_real.device(), T::DTYPE, y_real.shape())?,
        T::from_f64(1e-15),
    )?;
    let pi = generic_mul_scalar::<T>(
        &generic_ones::<T>(y_real.device(), T::DTYPE, y_real.shape())?,
        T::from_f64(std::f64::consts::PI),
    )?;
    let pi_half = generic_mul_scalar::<T>(&pi, T::from_f64(0.5))?;

    // Compute magnitudes
    let y_mag = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(y_real, 2.0)?,
            &generic_powf::<T>(y_imag, 2.0)?,
        )?,
        0.5,
    )?;
    let x_mag = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(x_real, 2.0)?,
            &generic_powf::<T>(x_imag, 2.0)?,
        )?,
        0.5,
    )?;

    // Special cases
    let both_zero = generic_mul::<T>(
        &generic_lt::<T>(&y_mag, &eps)?,
        &generic_lt::<T>(&x_mag, &eps)?,
    )?;
    let x_zero = generic_lt::<T>(&x_mag, &eps)?;

    // Base atan computation
    let (z_real, z_imag) = generic_complex_div::<T>(y_real, y_imag, x_real, x_imag)?;
    let (base_atan_real, base_atan_imag) = generic_complex_atan::<T>(&z_real, &z_imag)?;

    // Quadrant adjustments
    let x_neg = generic_lt::<T>(x_real, &zero)?;
    let y_gte_zero = generic_gte::<T>(y_real, &zero)?;

    // When x < 0: add π for y ≥ 0, subtract π for y < 0
    let adjustment = generic_where::<T>(
        &x_neg,
        &generic_where::<T>(&y_gte_zero, &pi, &generic_neg::<T>(&pi)?)?,
        &zero,
    )?;

    // Apply adjustment to real part only
    let adjusted_real = generic_add::<T>(&base_atan_real, &adjustment)?;

    // Handle x = 0 cases
    let x_zero_real = generic_where::<T>(&y_gte_zero, &pi_half, &generic_neg::<T>(&pi_half)?)?;
    let x_zero_imag = zero.clone();

    // Combine all cases
    let (real, imag) = generic_where_complex::<T>(
        &x_zero,
        &x_zero_real,
        &x_zero_imag,
        &adjusted_real,
        &base_atan_imag,
    )?;
    let (result_real, result_imag) =
        generic_where_complex::<T>(&both_zero, &zero, &zero, &real, &imag)?;

    Ok((result_real, result_imag))
}

// Generic function for complex exp
#[inline]
pub fn generic_complex_exp<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    let exp_real = generic_exp::<T>(real)?;
    Ok((
        generic_mul::<T>(&exp_real, &generic_cos::<T>(imag)?)?,
        generic_mul::<T>(&exp_real, &generic_sin::<T>(imag)?)?,
    ))
}

// Generic function for complex log
#[inline]
pub fn generic_complex_log<T: WithDType>(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    let abs = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let arg = generic_atan2::<T>(imag, real)?;
    Ok((generic_log::<T>(&abs)?, arg))
}

// Generic function for complex where
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
