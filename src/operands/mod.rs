// Macro for ElementWiseOp
macro_rules! impl_elementwise_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> ElementWiseOp<T> for $type<T> {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.add(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.sub(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.mul(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.div(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                Ok(Self(self.$tensor_field.clamp(*min, *max)?, PhantomData))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ElementWiseOp<T> for $type<T, $rows> {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.add(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.sub(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.mul(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.div(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                Ok(Self(self.$tensor_field.clamp(*min, *max)?, PhantomData))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ElementWiseOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.add(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.sub(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.mul(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    self.$tensor_field.div(&rhs.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                Ok(Self(self.$tensor_field.clamp(*min, *max)?, PhantomData))
            }
        }
    };
}

// Macro for ScalarOp
macro_rules! impl_scalar_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> ScalarOp<T> for $type<T> {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_add(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_sub(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_mul(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_div(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn powf(&self, exponent: f64) -> Result<Self> {
                Ok(Self(self.$tensor_field.powf(exponent)?, PhantomData))
            }
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self> {
                Ok(Self(
                    self.$tensor_field.pow(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn pow_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_pow(&scalar_tensor)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ScalarOp<T> for $type<T, $rows> {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_add(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_sub(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_mul(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_div(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn powf(&self, exponent: f64) -> Result<Self> {
                Ok(Self(self.$tensor_field.powf(exponent)?, PhantomData))
            }
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self> {
                Ok(Self(
                    self.$tensor_field.pow(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn pow_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_pow(&scalar_tensor)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ScalarOp<T>
            for $type<T, $rows, $cols>
        {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_add(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_sub(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_mul(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_div(&scalar_tensor)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn powf(&self, exponent: f64) -> Result<Self> {
                Ok(Self(self.$tensor_field.powf(exponent)?, PhantomData))
            }
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self> {
                Ok(Self(
                    self.$tensor_field.pow(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn pow_scalar(&self, scalar: T) -> Result<Self> {
                let scalar_tensor =
                    Tensor::from_vec(vec![scalar], (1, 1), self.$tensor_field.device())?;
                Ok(Self(
                    self.$tensor_field.broadcast_pow(&scalar_tensor)?,
                    PhantomData,
                ))
            }
        }
    };
}

// Macro for TrigOp
macro_rules! impl_trig_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> TrigOp<T> for $type<T> {
            type Output = Self;
            #[inline]
            fn sin(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.sin()?, PhantomData))
            }
            #[inline]
            fn cos(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.cos()?, PhantomData))
            }
            #[inline]
            fn sinh(&self) -> Result<Self::Output> {
                Ok(Self(
                    self.exp()?
                        .sub(&self.neg()?.exp()?)?
                        .div_scalar(T::from_f64(2.0))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn cosh(&self) -> Result<Self::Output> {
                Ok(Self(
                    self.exp()?
                        .add(&self.neg()?.exp()?)?
                        .div_scalar(T::from_f64(2.0))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.tanh()?, PhantomData))
            }
            #[inline]
            fn atan(&self) -> Result<Self::Output> {
                let i = Self::zeros(self.$tensor_field.device())?.add_scalar(T::from_f64(1.0))?;

                let numerator = i.add(self)?;
                let denominator = i.sub(self)?;

                Ok(Self(
                    numerator
                        .div(&denominator)?
                        .log()?
                        .mul_scalar(T::from_f64(0.5))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn atan2(&self, x: &Self) -> Result<Self::Output> {
                let zero = Self::zeros(self.$tensor_field.device())?;
                let eps =
                    Self::ones(self.$tensor_field.device())?.mul_scalar(T::from_f64(1e-15))?;
                let pi = Self::ones(self.$tensor_field.device())?.mul_scalar(T::from_f64(PI))?;
                let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

                // Compute magnitudes
                let y_mag = self.abs()?;
                let x_mag = x.abs()?;

                // Special cases
                let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
                let x_zero = x_mag.lt(&eps)?;

                // Base atan computation
                let base_atan = self.div(x)?.atan()?;

                // Quadrant adjustments
                let x_lt_zero = x.lt(&zero)?;
                let y_gte_zero = self.gte(&zero)?;
                let adjustment = x_lt_zero.and(&y_gte_zero)?.where_cond(&pi, &pi.neg()?)?;
                let adjusted_atan = base_atan.add(&adjustment)?;

                // Handle x = 0 cases
                let x_zero_result = y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?;

                // Combine all cases
                let result = both_zero
                    .where_cond(&zero, &x_zero.where_cond(&x_zero_result, &adjusted_atan)?)?;

                Ok(result)
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> TrigOp<T> for $type<T, $rows> {
            type Output = Self;
            #[inline]
            fn sin(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.sin()?, PhantomData))
            }
            #[inline]
            fn cos(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.cos()?, PhantomData))
            }
            #[inline]
            fn sinh(&self) -> Result<Self::Output> {
                Ok(Self(
                    self.exp()?
                        .sub(&self.neg()?.exp()?)?
                        .div_scalar(T::from_f64(2.0))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn cosh(&self) -> Result<Self::Output> {
                Ok(Self(
                    self.exp()?
                        .add(&self.neg()?.exp()?)?
                        .div_scalar(T::from_f64(2.0))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.tanh()?, PhantomData))
            }
            #[inline]
            fn atan(&self) -> Result<Self::Output> {
                let i = Self::zeros(self.$tensor_field.device())?.add_scalar(T::from_f64(1.0))?;

                let numerator = i.add(self)?;
                let denominator = i.sub(self)?;

                Ok(Self(
                    numerator
                        .div(&denominator)?
                        .log()?
                        .mul_scalar(T::from_f64(0.5))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn atan2(&self, x: &Self) -> Result<Self::Output> {
                let zero = Self::zeros(self.$tensor_field.device())?;
                let eps =
                    Self::ones(self.$tensor_field.device())?.mul_scalar(T::from_f64(1e-15))?;
                let pi = Self::ones(self.$tensor_field.device())?.mul_scalar(T::from_f64(PI))?;
                let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

                // Compute magnitudes
                let y_mag = self.abs()?;
                let x_mag = x.abs()?;

                // Special cases
                let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
                let x_zero = x_mag.lt(&eps)?;

                // Base atan computation
                let base_atan = self.div(x)?.atan()?;

                // Quadrant adjustments
                let x_lt_zero = x.lt(&zero)?;
                let y_gte_zero = self.gte(&zero)?;
                let adjustment = x_lt_zero.and(&y_gte_zero)?.where_cond(&pi, &pi.neg()?)?;
                let adjusted_atan = base_atan.add(&adjustment)?;

                // Handle x = 0 cases
                let x_zero_result = y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?;

                // Combine all cases
                let result = both_zero
                    .where_cond(&zero, &x_zero.where_cond(&x_zero_result, &adjusted_atan)?)?;

                Ok(result)
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TrigOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;
            #[inline]
            fn sin(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.sin()?, PhantomData))
            }
            #[inline]
            fn cos(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.cos()?, PhantomData))
            }
            #[inline]
            fn sinh(&self) -> Result<Self::Output> {
                Ok(Self(
                    self.exp()?
                        .sub(&self.neg()?.exp()?)?
                        .div_scalar(T::from_f64(2.0))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn cosh(&self) -> Result<Self::Output> {
                Ok(Self(
                    self.exp()?
                        .add(&self.neg()?.exp()?)?
                        .div_scalar(T::from_f64(2.0))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                Ok(Self(self.$tensor_field.tanh()?, PhantomData))
            }
            #[inline]
            fn atan(&self) -> Result<Self::Output> {
                let i = Self::zeros(self.$tensor_field.device())?.add_scalar(T::from_f64(1.0))?;

                let numerator = i.add(self)?;
                let denominator = i.sub(self)?;

                Ok(Self(
                    numerator
                        .div(&denominator)?
                        .log()?
                        .mul_scalar(T::from_f64(0.5))?
                        .0,
                    PhantomData,
                ))
            }
            #[inline]
            fn atan2(&self, x: &Self) -> Result<Self::Output> {
                let zero = Self::zeros(self.$tensor_field.device())?;
                let eps =
                    Self::ones(self.$tensor_field.device())?.mul_scalar(T::from_f64(1e-15))?;
                let pi = Self::ones(self.$tensor_field.device())?.mul_scalar(T::from_f64(PI))?;
                let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

                // Compute magnitudes
                let y_mag = self.abs()?;
                let x_mag = x.abs()?;

                // Special cases
                let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
                let x_zero = x_mag.lt(&eps)?;

                // Base atan computation
                let base_atan = self.div(x)?.atan()?;

                // Quadrant adjustments
                let x_lt_zero = x.lt(&zero)?;
                let y_gte_zero = self.gte(&zero)?;
                let adjustment = x_lt_zero.and(&y_gte_zero)?.where_cond(&pi, &pi.neg()?)?;
                let adjusted_atan = base_atan.add(&adjustment)?;

                // Handle x = 0 cases
                let x_zero_result = y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?;

                // Combine all cases
                let result = both_zero
                    .where_cond(&zero, &x_zero.where_cond(&x_zero_result, &adjusted_atan)?)?;

                Ok(result)
            }
        }
    };
}

// Macro for UnaryOp
macro_rules! impl_unary_op {
    ($type:ident, $tensor_field:tt, $transpose_output:ident) => {
        impl<T: WithDType> UnaryOp<T> for $type<T> {
            type TransposeOutput = $transpose_output<T>;
            type ScalarOutput = Scalar<T>;
            #[inline]
            fn neg(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.neg()?, PhantomData))
            }
            #[inline]
            fn abs(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.abs()?, PhantomData))
            }
            #[inline]
            fn exp(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.exp()?, PhantomData))
            }
            #[inline]
            fn log(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.log()?, PhantomData))
            }
            #[inline]
            fn mean(&self) -> Result<Self::ScalarOutput> {
                Ok(Scalar(self.$tensor_field.mean_all()?, PhantomData))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $transpose_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> UnaryOp<T> for $type<T, $rows> {
            type TransposeOutput = $transpose_output<T, $rows>;
            type ScalarOutput = Scalar<T>;
            #[inline]
            fn neg(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.neg()?, PhantomData))
            }
            #[inline]
            fn abs(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.abs()?, PhantomData))
            }
            #[inline]
            fn exp(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.exp()?, PhantomData))
            }
            #[inline]
            fn log(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.log()?, PhantomData))
            }
            #[inline]
            fn mean(&self) -> Result<Self::ScalarOutput> {
                Ok(Scalar(self.$tensor_field.mean_all()?, PhantomData))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $transpose_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> UnaryOp<T>
            for $type<T, $rows, $cols>
        {
            type TransposeOutput = $transpose_output<T, $cols, $rows>;
            type ScalarOutput = Scalar<T>;
            #[inline]
            fn neg(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.neg()?, PhantomData))
            }
            #[inline]
            fn abs(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.abs()?, PhantomData))
            }
            #[inline]
            fn exp(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.exp()?, PhantomData))
            }
            #[inline]
            fn log(&self) -> Result<Self> {
                Ok(Self(self.$tensor_field.log()?, PhantomData))
            }
            #[inline]
            fn mean(&self) -> Result<Self::ScalarOutput> {
                Ok(Scalar(self.$tensor_field.mean_all()?, PhantomData))
            }
        }
    };
}

// Macro for ComparisonOp
macro_rules! impl_comparison_op {
    ($type:ident, $tensor_field:tt, $output_type:ident) => {
        impl<T: WithDType> ComparisonOp<T> for $type<T> {
            type Output = $output_type<u8>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.lt(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.le(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.eq(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.ne(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.gt(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.ge(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $output_type:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ComparisonOp<T> for $type<T, $rows> {
            type Output = $output_type<u8, $rows>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.lt(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.le(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.eq(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.ne(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.gt(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.ge(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $output_type:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ComparisonOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = $output_type<u8, $rows, $cols>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.lt(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.le(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.eq(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.ne(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.gt(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    self.$tensor_field.ge(&other.$tensor_field)?,
                    PhantomData,
                ))
            }
        }
    };
}

// Macro for ComplexOp
macro_rules! impl_complex_op {
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident) => {
        impl<T: WithDType> ComplexOp<T> for $type<T> {
            type Output = Self;
            type RealOutput = $real_output<T>;
            #[inline]
            fn conj(&self) -> Result<Self::Output> {
                Ok(Self {
                    real: self.real.clone(),
                    imag: self.imag.neg()?,
                })
            }
            #[inline]
            fn magnitude(&self) -> Result<Self::RealOutput> {
                Ok($real_output(
                    self.$real_field
                        .0
                        .sqr()?
                        .add(&self.$imag_field.0.sqr()?)?
                        .sqrt()?,
                    PhantomData,
                ))
            }
            #[inline]
            fn arg(&self) -> Result<Self::RealOutput> {
                self.imag.atan2(&self.real)
            }
            #[inline]
            fn real(&self) -> Result<Self::RealOutput> {
                Ok(self.real.clone())
            }
            #[inline]
            fn imaginary(&self) -> Result<Self::RealOutput> {
                Ok(self.imag.clone())
            }
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self::Output> {
                self.log()?.mul(other)?.exp()
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ComplexOp<T> for $type<T, $rows> {
            type Output = Self;
            type RealOutput = $real_output<T, $rows>;
            #[inline]
            fn conj(&self) -> Result<Self::Output> {
                Ok(Self {
                    real: self.real.clone(),
                    imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
                })
            }
            #[inline]
            fn magnitude(&self) -> Result<Self::RealOutput> {
                Ok($real_output::<T, $rows>(
                    self.$real_field
                        .0
                        .sqr()?
                        .add(&self.$imag_field.0.sqr()?)?
                        .sqrt()?,
                    PhantomData,
                ))
            }
            #[inline]
            fn arg(&self) -> Result<Self::RealOutput> {
                self.imag.atan2(&self.real)
            }
            #[inline]
            fn real(&self) -> Result<Self::RealOutput> {
                Ok(self.real.clone())
            }
            #[inline]
            fn imaginary(&self) -> Result<Self::RealOutput> {
                Ok(self.imag.clone())
            }
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self::Output> {
                self.log()?.mul(other)?.exp()
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ComplexOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;
            type RealOutput = $real_output<T, $rows, $cols>;
            #[inline]
            fn conj(&self) -> Result<Self::Output> {
                Ok(Self {
                    real: self.real.clone(),
                    imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
                })
            }
            #[inline]
            fn magnitude(&self) -> Result<Self::RealOutput> {
                Ok($real_output::<T, $rows, $cols>(
                    self.$real_field
                        .0
                        .sqr()?
                        .add(&self.$imag_field.0.sqr()?)?
                        .sqrt()?,
                    PhantomData,
                ))
            }
            #[inline]
            fn arg(&self) -> Result<Self::RealOutput> {
                self.imag.atan2(&self.real)
            }
            #[inline]
            fn real(&self) -> Result<Self::RealOutput> {
                Ok(self.real.clone())
            }
            #[inline]
            fn imaginary(&self) -> Result<Self::RealOutput> {
                Ok(self.imag.clone())
            }
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self::Output> {
                self.log()?.mul(other)?.exp()
            }
        }
    };
}

use crate::TensorBase;
use candle_core::WithDType;
pub use {
    column_vector::ColumnVector, complex_column_vector::ComplexColumnVector,
    complex_matrix::ComplexMatrix, complex_row_vector::ComplexRowVector,
    complex_scalar::ComplexScalar, matrix::Matrix, real_scaler::Scalar, row_vector::RowVector,
};

// pub fn generic_atan2<T: WithDType>(
//     y: &impl TensorBase<T>,
//     x: &impl TensorBase<T>,
// ) -> Result<impl TensorBase<T>> {
//     let zero = T::zeros(y.device())?;
//     let eps = T::ones(y.device())?.mul_scalar(T::from_f64(1e-15))?;
//     let pi = T::ones(y.device())?.mul_scalar(T::from_f64(PI))?;
//     let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

//     // Compute magnitudes
//     let y_mag = y.abs()?;
//     let x_mag = x.abs()?;

//     // Special cases
//     let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
//     let x_zero = x_mag.lt(&eps)?;

//     // Base atan computation
//     let base_atan = y.div(x)?.atan()?;

//     // Quadrant adjustments
//     let x_lt_zero = x.lt(&zero)?;
//     let y_gte_zero = y.gte(&zero)?;
//     let adjustment = x_lt_zero.where_cond(&y_gte_zero.where_cond(&pi, &pi.neg()?)?, &zero)?;
//     let adjusted_atan = base_atan.add(&adjustment)?;

//     // Handle x = 0 cases
//     let x_zero_result = y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?;

//     // Combine all cases
//     let result =
//         both_zero.where_cond(&zero, &x_zero.where_cond(&x_zero_result, &adjusted_atan)?)?;

//     Ok(result)
// }

mod column_vector;
mod complex_column_vector;
mod complex_matrix;
mod complex_row_vector;
mod complex_scalar;
mod matrix;
mod real_scaler;
mod row_vector;

#[cfg(test)]
mod test {
    use approx::{assert_relative_eq, RelativeEq};
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
}
