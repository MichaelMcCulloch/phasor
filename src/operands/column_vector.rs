use crate::*;
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::{f64::consts::PI, marker::PhantomData};

#[derive(Debug, Clone)]
pub struct ColumnVector<T: WithDType, const ROWS: usize>(
    pub(crate) Tensor,
    pub(crate) PhantomData<T>,
);
impl<T: WithDType, const ROWS: usize> IsColumnVector<ROWS> for ColumnVector<T, ROWS> {}
impl<T: WithDType, const ROWS: usize> ColumnVector<T, ROWS> {
    pub fn new(data: &[T], device: &Device) -> Result<ColumnVector<T, ROWS>> {
        assert!(data.len() == ROWS);
        Ok(Self(
            Tensor::from_slice(data, (ROWS, 1), device)?,
            PhantomData,
        ))
    }
}
impl<T: WithDType, const ROWS: usize> TensorBase<T> for ColumnVector<T, ROWS> {
    type ReadOutput = Vec<T>;
    #[inline]
    fn device(&self) -> &Device {
        self.0.device()
    }
    #[inline]
    fn dtype() -> DType {
        T::DTYPE
    }
    #[inline]
    fn shape() -> (usize, usize) {
        (ROWS, 1)
    }
    #[inline]
    fn read(&self) -> Result<Self::ReadOutput> {
        Ok(self.0.to_vec2()?.into_iter().flatten().collect())
    }
}
impl<T: WithDType, const ROWS: usize> ElementWiseOp<T> for ColumnVector<T, ROWS> {
    type Output = ColumnVector<T, ROWS>;
    #[inline]
    fn add(&self, rhs: &Self) -> Result<Self::Output> {
        Ok(Self(self.0.add(&rhs.0)?, PhantomData))
    }
    #[inline]
    fn sub(&self, rhs: &Self) -> Result<Self::Output> {
        Ok(Self(self.0.sub(&rhs.0)?, PhantomData))
    }
    #[inline]
    fn mul(&self, rhs: &Self) -> Result<Self::Output> {
        Ok(Self(self.0.mul(&rhs.0)?, PhantomData))
    }
    #[inline]
    fn div(&self, rhs: &Self) -> Result<Self::Output> {
        Ok(Self(self.0.div(&rhs.0)?, PhantomData))
    }
    #[inline]
    fn clamp(&self, min: &T, max: &T) -> Result<Self> {
        Ok(Self(self.0.clamp(*min, *max)?, PhantomData))
    }
}
impl<T: WithDType, const ROWS: usize> ScalarOp<T> for ColumnVector<T, ROWS> {
    #[inline]
    fn add_scalar(&self, scalar: T) -> Result<Self> {
        let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), self.0.device())?;
        Ok(Self(self.0.broadcast_add(&scalar_tensor)?, PhantomData))
    }
    #[inline]
    fn sub_scalar(&self, scalar: T) -> Result<Self> {
        let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), self.0.device())?;
        Ok(Self(self.0.broadcast_sub(&scalar_tensor)?, PhantomData))
    }
    #[inline]
    fn mul_scalar(&self, scalar: T) -> Result<Self> {
        let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), self.0.device())?;
        Ok(Self(self.0.broadcast_mul(&scalar_tensor)?, PhantomData))
    }
    #[inline]
    fn div_scalar(&self, scalar: T) -> Result<Self> {
        let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), self.0.device())?;
        Ok(Self(self.0.broadcast_div(&scalar_tensor)?, PhantomData))
    }
    #[inline]
    fn powf(&self, exponent: f64) -> Result<Self> {
        Ok(Self(self.0.powf(exponent)?, PhantomData))
    }
    #[inline]
    fn pow(&self, other: &Self) -> Result<Self> {
        Ok(Self(self.0.pow(&other.0)?, PhantomData))
    }
    #[inline]
    fn pow_scalar(&self, scalar: T) -> Result<Self> {
        let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), self.0.device())?;
        Ok(Self(self.0.broadcast_pow(&scalar_tensor)?, PhantomData))
    }
}
impl<T: WithDType, const ROWS: usize> TrigOp<T> for ColumnVector<T, ROWS> {
    type Output = ColumnVector<T, ROWS>;
    #[inline]
    fn sin(&self) -> Result<Self::Output> {
        Ok(Self(self.0.sin()?, PhantomData))
    }
    #[inline]
    fn cos(&self) -> Result<Self::Output> {
        Ok(Self(self.0.cos()?, PhantomData))
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
        Ok(Self(self.0.tanh()?, PhantomData))
    }
    #[inline]
    fn atan(&self) -> Result<Self::Output> {
        let i = Self::zeros(self.0.device())?.add_scalar(T::from_f64(1.0))?;

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
        let zero = Self::zeros(self.0.device())?;
        let eps = Self::ones(self.0.device())?.mul_scalar(T::from_f64(1e-15))?;
        let pi = Self::ones(self.0.device())?.mul_scalar(T::from_f64(PI))?;
        let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

        // Compute magnitudes
        let y_mag = self.abs()?;
        let x_mag = x.abs()?;

        // Special cases
        let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
        let x_zero = x_mag.lt(&eps)?;

        // Base atan computation
        let z = self.div(x)?;
        let base_atan = z.atan()?;

        // Quadrant adjustments
        let x_neg = x.lt(&zero)?;
        let y_gte_zero = self.gte(&zero)?;

        // When x < 0: add π for y ≥ 0, subtract π for y < 0
        let adjustment = x_neg.where_cond(&y_gte_zero.where_cond(&pi, &pi.neg()?)?, &zero)?;

        // Apply adjustment to real part only
        let adjusted_result = base_atan.add(&adjustment)?;

        // Handle x = 0 cases
        let x_zero_result = y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?;

        // Combine all cases
        let result =
            both_zero.where_cond(&zero, &x_zero.where_cond(&x_zero_result, &adjusted_result)?)?;

        Ok(result)
    }
}
impl<T: WithDType, const ROWS: usize> UnaryOp<T> for ColumnVector<T, ROWS> {
    type TransposeOutput = RowVector<T, ROWS>;
    type ScalarOutput = Scalar<T>;
    #[inline]
    fn neg(&self) -> Result<Self> {
        Ok(Self(self.0.neg()?, PhantomData))
    }
    #[inline]
    fn abs(&self) -> Result<Self> {
        Ok(Self(self.0.abs()?, PhantomData))
    }
    #[inline]
    fn exp(&self) -> Result<Self> {
        Ok(Self(self.0.exp()?, PhantomData))
    }
    #[inline]
    fn log(&self) -> Result<Self> {
        Ok(Self(self.0.log()?, PhantomData))
    }
    #[inline]
    fn mean(&self) -> Result<Self::ScalarOutput> {
        Ok(Scalar(self.0.mean_all()?, PhantomData))
    }
}
impl<T: WithDType, const ROWS: usize> ComparisonOp<T> for ColumnVector<T, ROWS> {
    type Output = ColumnVector<u8, ROWS>;
    #[inline]
    fn lt(&self, other: &Self) -> Result<Self::Output> {
        Ok(ColumnVector(self.0.lt(&other.0)?, PhantomData))
    }
    #[inline]
    fn lte(&self, other: &Self) -> Result<Self::Output> {
        Ok(ColumnVector(self.0.le(&other.0)?, PhantomData))
    }
    #[inline]
    fn eq(&self, other: &Self) -> Result<Self::Output> {
        Ok(ColumnVector(self.0.eq(&other.0)?, PhantomData))
    }
    #[inline]
    fn ne(&self, other: &Self) -> Result<Self::Output> {
        Ok(ColumnVector(self.0.ne(&other.0)?, PhantomData))
    }
    #[inline]
    fn gt(&self, other: &Self) -> Result<Self::Output> {
        Ok(ColumnVector(self.0.gt(&other.0)?, PhantomData))
    }
    #[inline]
    fn gte(&self, other: &Self) -> Result<Self::Output> {
        Ok(ColumnVector(self.0.ge(&other.0)?, PhantomData))
    }
}
impl<T: WithDType, const ROWS: usize> TensorFactory<T> for ColumnVector<T, ROWS> {
    #[inline]
    fn zeros(device: &Device) -> Result<Self> {
        Ok(Self(
            Tensor::zeros(Self::shape(), T::DTYPE, device)?,
            PhantomData,
        ))
    }
    #[inline]
    fn ones(device: &Device) -> Result<Self> {
        Ok(Self(
            Tensor::ones(Self::shape(), T::DTYPE, device)?,
            PhantomData,
        ))
    }
    #[inline]
    fn ones_neg(device: &Device) -> Result<Self> {
        Ok(Self(
            Tensor::ones(Self::shape(), T::DTYPE, device)?.neg()?,
            PhantomData,
        ))
    }
}
impl<F: FloatDType, const ROWS: usize> TensorFactoryFloat<F> for ColumnVector<F, ROWS> {
    #[inline]
    fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
        Ok(Self(
            Tensor::randn(mean, std, Self::shape(), device)?,
            PhantomData,
        ))
    }
    #[inline]
    fn randu(low: F, high: F, device: &Device) -> Result<Self> {
        Ok(Self(
            Tensor::rand(low, high, Self::shape(), device)?,
            PhantomData,
        ))
    }
}
impl<T: WithDType, const ROWS: usize> ConditionalOp<T> for ColumnVector<u8, ROWS> {
    type Output = ColumnVector<T, ROWS>;
    type ComplexOutput = ComplexColumnVector<T, ROWS>;
    #[inline]
    fn where_cond(&self, on_true: &Self::Output, on_false: &Self::Output) -> Result<Self::Output> {
        Ok(ColumnVector::<T, ROWS>(
            self.0.where_cond(&on_true.0, &on_false.0)?,
            PhantomData,
        ))
    }
    #[inline]
    fn where_cond_complex(
        &self,
        on_true: &Self::ComplexOutput,
        on_false: &Self::ComplexOutput,
    ) -> Result<Self::ComplexOutput> {
        Ok(ComplexColumnVector::<T, ROWS> {
            real: self.where_cond(&on_true.real()?, &on_false.real()?)?,
            imag: self.where_cond(&on_true.imaginary()?, &on_false.imaginary()?)?,
        })
    }
    #[inline]
    fn promote(&self, dtype: DType) -> Result<Self::Output> {
        Ok(ColumnVector(self.0.to_dtype(dtype)?, PhantomData))
    }
}

impl<const ROWS: usize> BooleanOp for ColumnVector<u8, ROWS> {
    type Output = ColumnVector<u8, ROWS>;

    #[inline]
    fn and(&self, other: &Self) -> Result<Self::Output> {
        Ok(Self(self.0.mul(&other.0)?, PhantomData))
    }

    #[inline]
    fn or(&self, other: &Self) -> Result<Self::Output> {
        Ok(Self(
            self.0.ne(0u8)?.add(&other.0.ne(0u8)?)?.ne(0u8)?,
            PhantomData,
        ))
    }

    #[inline]
    fn xor(&self, other: &Self) -> Result<Self::Output> {
        Ok(Self(self.0.ne(&other.0)?, PhantomData))
    }

    #[inline]
    fn not(&self) -> Result<Self::Output> {
        Ok(Self(self.0.eq(0u8)?, PhantomData))
    }
}
impl<T: WithDType, const ROWS: usize> ColumnVectorOps<T, ROWS> for ColumnVector<T, ROWS> {
    type TransposeOutput = RowVector<T, ROWS>;
    type OuterInput<const COLS: usize> = RowVector<T, COLS>;
    type OuterOutput<const COLS: usize> = Matrix<T, ROWS, COLS>;
    type BroadcastOutput<const C: usize> = Matrix<T, ROWS, C>;
    #[inline]
    fn outer<const COLS: usize>(
        &self,
        other: &Self::OuterInput<COLS>,
    ) -> Result<Self::OuterOutput<COLS>> {
        Ok(Matrix(self.0.matmul(&other.0)?, PhantomData))
    }
    #[inline]
    fn transpose(&self) -> Result<Self::TransposeOutput> {
        Ok(RowVector(self.0.t()?, PhantomData))
    }
    #[inline]
    fn broadcast<const C: usize>(&self) -> Result<Self::BroadcastOutput<C>> {
        Ok(Matrix(self.0.broadcast_as((ROWS, C))?, PhantomData))
    }
}
