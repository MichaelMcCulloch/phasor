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
