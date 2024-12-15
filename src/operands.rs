mod scalar {
    use crate::ops::*;
    use crate::{
        ColumnVector, ComplexColumnVector, ComplexMatrix, ComplexRowVector, ComplexScalar,
        RowVector,
    };
    use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
    use std::{f64::consts::PI, marker::PhantomData};
    #[derive(Debug, Clone)]
    pub struct Scalar<T: WithDType>(pub(crate) Tensor, pub(crate) PhantomData<T>);
    impl<T: WithDType> Scalar<T> {
        fn adjust_quadrant(
            &self,
            base_atan: Self,
            x: &Self,
            zero: &Self,
            pi: &Self,
        ) -> Result<Self> {
            let x_lt_zero = x.lt(zero)?;
            let y_gte_zero = self.gte(zero)?;
            let adjustment = x_lt_zero.and(&y_gte_zero)?.where_cond(pi, &pi.neg()?)?;
            base_atan.add(&adjustment)
        }

        fn handle_x_zero(&self, x: &Self, zero: &Self, pi_half: &Self) -> Result<Self> {
            let y_gte_zero = self.gte(zero)?;
            y_gte_zero.where_cond(pi_half, &pi_half.neg()?)
        }
    }
    impl<T: WithDType> IsScalar for Scalar<T> {}
    impl<T: WithDType> TensorBase<T> for Scalar<T> {
        type ReadOutput = T;
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
            (1, 1)
        }
        #[inline]
        fn read(&self) -> Result<Self::ReadOutput> {
            Ok(self
                .0
                .reshape((1, 1))?
                .to_vec2()?
                .into_iter()
                .flatten()
                .last()
                .unwrap())
        }
    }
    impl<T: WithDType> Scalar<T> {
        pub fn new(data: T, device: &Device) -> Result<Scalar<T>> {
            Ok(Self(
                Tensor::from_slice(&[data], Self::shape(), device)?,
                PhantomData,
            ))
        }
    }
    impl<T: WithDType> ElementWiseOp<T> for Scalar<T> {
        type Output = Scalar<T>;
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
            Ok(Scalar(self.0.clamp(*min, *max)?, PhantomData))
        }
    }
    impl<T: WithDType> ScalarOp<T> for Scalar<T> {
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
    impl<T: WithDType> TrigOp<T> for Scalar<T> {
        type Output = Scalar<T>;
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
            let one = Self::ones(self.0.device())?;
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
            let base_atan = self.div(x)?.atan()?;

            // Quadrant adjustments
            let adjusted_atan = self.adjust_quadrant(base_atan, x, &zero, &pi)?;

            // Handle x = 0 cases
            let x_zero_result = self.handle_x_zero(x, &zero, &pi_half)?;

            // Combine all cases
            let result =
                both_zero.where_cond(&zero, &x_zero.where_cond(&x_zero_result, &adjusted_atan)?)?;

            Ok(result)
        }
    }
    impl<T: WithDType> UnaryOp<T> for Scalar<T> {
        type ScalarOutput = Scalar<T>;
        type TransposeOutput = Scalar<T>;
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
            Ok(self.clone())
        }
    }
    impl<T: WithDType> TensorFactory<T> for Scalar<T> {
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
    impl<T: WithDType> ConditionalOp<T> for Scalar<u8> {
        type Output = Scalar<T>;
        type ComplexOutput = ComplexScalar<T>;

        #[inline]
        fn where_cond(
            &self,
            on_true: &Self::Output,
            on_false: &Self::Output,
        ) -> Result<Self::Output> {
            Ok(Scalar::<T>(
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
            Ok(ComplexScalar::<T> {
                real: self.where_cond(&on_true.real()?, &on_false.real()?)?,
                imag: self.where_cond(&on_true.imaginary()?, &on_false.imaginary()?)?,
            })
        }
        #[inline]
        fn promote(&self, dtype: DType) -> Result<Self::Output> {
            Ok(Scalar(self.0.to_dtype(dtype)?, PhantomData))
        }
    }

    impl BooleanOp for Scalar<u8> {
        type Output = Scalar<u8>;
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
    impl<F: FloatDType> TensorFactoryFloat<F> for Scalar<F> {
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
    impl<T: WithDType> ComparisonOp<T> for Scalar<T> {
        type Output = Scalar<u8>;
        #[inline]
        fn lt(&self, other: &Self) -> Result<Self::Output> {
            Ok(Scalar(self.0.lt(&other.0)?, PhantomData))
        }
        #[inline]
        fn lte(&self, other: &Self) -> Result<Self::Output> {
            Ok(Scalar(self.0.le(&other.0)?, PhantomData))
        }
        #[inline]
        fn eq(&self, other: &Self) -> Result<Self::Output> {
            Ok(Scalar(self.0.eq(&other.0)?, PhantomData))
        }
        #[inline]
        fn ne(&self, other: &Self) -> Result<Self::Output> {
            Ok(Scalar(self.0.ne(&other.0)?, PhantomData))
        }
        #[inline]
        fn gt(&self, other: &Self) -> Result<Self::Output> {
            Ok(Scalar(self.0.gt(&other.0)?, PhantomData))
        }
        #[inline]
        fn gte(&self, other: &Self) -> Result<Self::Output> {
            Ok(Scalar(self.0.ge(&other.0)?, PhantomData))
        }
    }

    #[cfg(test)]
    mod test {
        use crate::*;
        use candle_core::{Device, Result};

        #[test]
        fn scalar_logical_and() -> Result<()> {
            let device = Device::Cpu;
            let a = Scalar::<u8>::new(1, &device)?;
            let b = Scalar::<u8>::new(0, &device)?;
            let c = Scalar::<u8>::new(1, &device)?;

            assert_eq!(a.and(&a)?.read()?, 1);
            assert_eq!(a.and(&b)?.read()?, 0);
            assert_eq!(b.and(&a)?.read()?, 0);
            assert_eq!(b.and(&b)?.read()?, 0);
            assert_eq!(a.and(&c)?.read()?, 1);

            Ok(())
        }

        #[test]
        fn scalar_logical_or() -> Result<()> {
            let device = Device::Cpu;
            let a = Scalar::<u8>::new(1, &device)?;
            let b = Scalar::<u8>::new(0, &device)?;
            let c = Scalar::<u8>::new(1, &device)?;

            assert_eq!(a.or(&a)?.read()?, 1);
            assert_eq!(a.or(&b)?.read()?, 1);
            assert_eq!(b.or(&a)?.read()?, 1);
            assert_eq!(b.or(&b)?.read()?, 0);
            assert_eq!(a.or(&c)?.read()?, 1);
            // Test overflow case
            let x: Scalar<u8> = Scalar::<u8>::new(255, &device)?;
            let y = Scalar::<u8>::new(1, &device)?;
            assert_eq!(x.or(&y)?.read()?, 1);

            Ok(())
        }

        #[test]
        fn scalar_logical_xor() -> Result<()> {
            let device = Device::Cpu;
            let a = Scalar::<u8>::new(1, &device)?;
            let b = Scalar::<u8>::new(0, &device)?;
            let c = Scalar::<u8>::new(1, &device)?;

            assert_eq!(a.xor(&a)?.read()?, 0);
            assert_eq!(a.xor(&b)?.read()?, 1);
            assert_eq!(b.xor(&a)?.read()?, 1);
            assert_eq!(b.xor(&b)?.read()?, 0);
            assert_eq!(a.xor(&c)?.read()?, 0);

            Ok(())
        }

        #[test]
        fn scalar_logical_not() -> Result<()> {
            let device = Device::Cpu;
            let a = Scalar::<u8>::new(1, &device)?;
            let b = Scalar::<u8>::new(0, &device)?;

            assert_eq!(a.not()?.read()?, 0);
            assert_eq!(b.not()?.read()?, 1);

            Ok(())
        }
    }
}
mod row_vector {
    use crate::ops::*;
    use crate::*;
    use crate::{ColumnVector, ComplexRowVector};
    use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
    use std::{f64::consts::PI, marker::PhantomData};
    #[derive(Debug, Clone)]
    pub struct RowVector<T: WithDType, const ROWS: usize>(
        pub(crate) Tensor,
        pub(crate) PhantomData<T>,
    );
    impl<T: WithDType, const ROWS: usize> IsRowVector<ROWS> for RowVector<T, ROWS> {}
    impl<T: WithDType, const COLS: usize> TensorBase<T> for RowVector<T, COLS> {
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
            (1, COLS)
        }
        #[inline]
        fn read(&self) -> Result<Self::ReadOutput> {
            Ok(self.0.to_vec2()?.into_iter().flatten().collect())
        }
    }
    impl<T: WithDType, const COLS: usize> RowVector<T, COLS> {
        pub fn new(data: &[T], device: &Device) -> Result<RowVector<T, COLS>> {
            assert!(data.len() == COLS);
            Ok(Self(
                Tensor::from_slice(data, (1, COLS), device)?,
                PhantomData,
            ))
        }
    }
    impl<T: WithDType, const COLS: usize> ElementWiseOp<T> for RowVector<T, COLS> {
        type Output = RowVector<T, COLS>;
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
    impl<T: WithDType, const COLS: usize> ScalarOp<T> for RowVector<T, COLS> {
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
    impl<T: WithDType, const COLS: usize> TrigOp<T> for RowVector<T, COLS> {
        type Output = RowVector<T, COLS>;
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
            let one = Self::ones(self.0.device())?;
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
            let ratio = self.div(x)?;
            let theta = ratio.atan()?;
            let x_lt_zero = x.lt(&zero)?;
            let y_gte_zero = self.gte(&zero)?;
            let pi = Self::ones(self.0.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_neg = Self::ones(self.0.device())?.mul_scalar(T::from_f64(-PI))?;
            let adjustment = x_lt_zero.where_cond(&y_gte_zero.where_cond(&pi, &pi_neg)?, &zero)?;
            theta.add(&adjustment)
        }
    }
    impl<T: WithDType, const COLS: usize> UnaryOp<T> for RowVector<T, COLS> {
        type TransposeOutput = Matrix<T, 1, COLS>;
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
    impl<T: WithDType, const COLS: usize> RowVectorOps<T, COLS> for RowVector<T, COLS> {
        type DotInput = RowVector<T, COLS>;
        type DotOutput = Scalar<T>;
        type MatMulMatrix<const M: usize> = Matrix<T, COLS, M>;
        type MatMulOutput<const M: usize> = RowVector<T, M>;
        type TransposeOutput = ColumnVector<T, COLS>;
        type BroadcastOutput<const ROWS: usize> = Matrix<T, ROWS, COLS>;
        #[inline]
        fn dot(&self, other: &Self::DotInput) -> Result<Self::DotOutput> {
            Ok(Scalar(
                self.0.mul(&other.0)?.sum_all()?.reshape((1, 1))?,
                PhantomData,
            ))
        }
        #[inline]
        fn matmul<const M: usize>(
            &self,
            other: &Self::MatMulMatrix<M>,
        ) -> Result<Self::MatMulOutput<M>> {
            Ok(RowVector(self.0.matmul(&other.0)?, PhantomData))
        }
        #[inline]
        fn transpose(&self) -> Result<Self::TransposeOutput> {
            Ok(ColumnVector(self.0.t()?, PhantomData))
        }
        #[inline]
        fn broadcast<const ROWS: usize>(&self) -> Result<Self::BroadcastOutput<ROWS>> {
            Ok(Matrix(self.0.broadcast_as((ROWS, COLS))?, PhantomData))
        }
    }
    impl<T: WithDType, const COLS: usize> ComparisonOp<T> for RowVector<T, COLS> {
        type Output = RowVector<u8, COLS>;
        #[inline]
        fn lt(&self, other: &Self) -> Result<Self::Output> {
            Ok(RowVector(self.0.lt(&other.0)?, PhantomData))
        }
        #[inline]
        fn lte(&self, other: &Self) -> Result<Self::Output> {
            Ok(RowVector(self.0.le(&other.0)?, PhantomData))
        }
        #[inline]
        fn eq(&self, other: &Self) -> Result<Self::Output> {
            Ok(RowVector(self.0.eq(&other.0)?, PhantomData))
        }
        #[inline]
        fn ne(&self, other: &Self) -> Result<Self::Output> {
            Ok(RowVector(self.0.ne(&other.0)?, PhantomData))
        }
        #[inline]
        fn gt(&self, other: &Self) -> Result<Self::Output> {
            Ok(RowVector(self.0.gt(&other.0)?, PhantomData))
        }
        #[inline]
        fn gte(&self, other: &Self) -> Result<Self::Output> {
            Ok(RowVector(self.0.ge(&other.0)?, PhantomData))
        }
    }
    impl<T: WithDType, const COLS: usize> TensorFactory<T> for RowVector<T, COLS> {
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
    impl<T: WithDType, const COLS: usize> ConditionalOp<T> for RowVector<u8, COLS> {
        type Output = RowVector<T, COLS>;
        type ComplexOutput = ComplexRowVector<T, COLS>;
        #[inline]
        fn where_cond(
            &self,
            on_true: &Self::Output,
            on_false: &Self::Output,
        ) -> Result<Self::Output> {
            Ok(RowVector::<T, COLS>(
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
            Ok(ComplexRowVector::<T, COLS> {
                real: RowVector(
                    self.0.where_cond(&on_true.real.0, &on_false.real.0)?,
                    PhantomData,
                ),
                imag: RowVector(
                    self.0.where_cond(&on_true.imag.0, &on_false.imag.0)?,
                    PhantomData,
                ),
            })
        }
        #[inline]
        fn promote(&self, dtype: DType) -> Result<Self::Output> {
            Ok(RowVector(self.0.to_dtype(dtype)?, PhantomData))
        }
    }
    impl<const COLS: usize> BooleanOp for RowVector<u8, COLS> {
        type Output = RowVector<u8, COLS>;

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
    impl<F: FloatDType, const COLS: usize> TensorFactoryFloat<F> for RowVector<F, COLS> {
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
    #[cfg(test)]
    mod test {

        use crate::*;
        use approx::assert_relative_eq;
        use candle_core::{DType, Device, Result};
        use operands::test::assert_relative_eq_vec;

        #[test]
        fn new_tensor1d() -> Result<()> {
            let device = Device::Cpu;
            let data: Vec<f64> = vec![1.0, 2.0, 3.0];
            let tensor = RowVector::<f64, 3>::new(&data, &device)?;
            assert_eq!(tensor.0.shape().dims(), [1, 3]);
            Ok(())
        }

        #[test]
        fn zeros() -> Result<()> {
            let device = Device::Cpu;
            let zeros = RowVector::<f64, 3>::zeros(&device)?;
            assert_eq!(zeros.read()?, vec![0.0, 0.0, 0.0]);
            Ok(())
        }

        #[test]
        fn ones() -> Result<()> {
            let device = Device::Cpu;
            let ones = RowVector::<f64, 3>::ones(&device)?;
            assert_eq!(ones.read()?, vec![1.0, 1.0, 1.0]);
            Ok(())
        }

        #[test]
        fn ones_neg() -> Result<()> {
            let device = Device::Cpu;
            let ones_neg = RowVector::<f64, 3>::ones_neg(&device)?;
            assert_eq!(ones_neg.read()?, vec![-1.0, -1.0, -1.0]);
            Ok(())
        }

        #[test]
        fn add_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let added = tensor.add_scalar(2.0)?;
            assert_eq!(added.read()?, vec![3.0, 4.0, 5.0]);
            Ok(())
        }

        #[test]
        fn sub_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let subbed = tensor.sub_scalar(1.0)?;
            assert_eq!(subbed.read()?, vec![0.0, 1.0, 2.0]);
            Ok(())
        }

        #[test]
        fn mul_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let mulled = tensor.mul_scalar(3.0)?;
            assert_eq!(mulled.read()?, vec![3.0, 6.0, 9.0]);
            Ok(())
        }

        #[test]
        fn div_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let divided = tensor.div_scalar(2.0)?;
            assert_eq!(divided.read()?, vec![0.5, 1.0, 1.5]);
            Ok(())
        }

        #[test]
        fn pow_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let powered = tensor.pow_scalar(2.0)?;
            assert_eq!(powered.read()?, vec![1.0, 4.0, 9.0]);
            Ok(())
        }

        #[test]
        fn element_wise_add() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
            let added = tensor1.add(&tensor2)?;
            assert_eq!(added.read()?, vec![3.0, 5.0, 7.0]);
            Ok(())
        }

        #[test]
        fn element_wise_sub() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
            let subbed = tensor1.sub(&tensor2)?;
            assert_eq!(subbed.read()?, vec![-1.0, -1.0, -1.0]);
            Ok(())
        }

        #[test]
        fn element_wise_mul() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
            let mulled = tensor1.mul(&tensor2)?;
            assert_eq!(mulled.read()?, vec![2.0, 6.0, 12.0]);
            Ok(())
        }

        #[test]
        fn element_wise_div() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
            let divided = tensor1.div(&tensor2)?;
            assert_eq!(divided.read()?, vec![0.5, 2.0 / 3.0, 0.75]);
            Ok(())
        }

        #[test]
        fn exp() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let exp = tensor.exp()?;
            let exp_vec = exp.read()?;
            assert_relative_eq!(exp_vec[0], 2.7182817);
            Ok(())
        }

        #[test]
        fn log() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let log = tensor.log()?;
            let log_vec = log.read()?;
            assert_relative_eq!(log_vec[0], 1.0f64.ln());
            assert_relative_eq!(log_vec[1], 2.0f64.ln());
            assert_relative_eq!(log_vec[2], 3.0f64.ln());
            Ok(())
        }

        #[test]
        fn cos() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let cos = tensor.cos()?;
            let cos_vec = cos.read()?;
            assert_relative_eq!(cos_vec[0], 0.5403023);
            Ok(())
        }
        #[test]
        fn sin() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let sin = tensor.sin()?;
            let sin_vec = sin.read()?;
            assert_relative_eq!(sin_vec[0], 0.8414709);
            Ok(())
        }

        #[test]
        fn dot_product() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
            let dot = tensor1.dot(&tensor2)?;
            assert_relative_eq!(dot.read()?, 20.0);
            Ok(())
        }

        #[test]
        fn outer_product() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = ColumnVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
            let outer = tensor1.outer(&tensor2)?;
            let outer_vec = outer.read()?;
            assert_eq!(outer_vec[0], vec![2.0, 3.0, 4.0]);
            assert_eq!(outer_vec[1], vec![4.0, 6.0, 8.0]);
            assert_eq!(outer_vec[2], vec![6.0, 9.0, 12.0]);
            Ok(())
        }

        #[test]
        fn where_condition() -> Result<()> {
            let device = Device::Cpu;
            let cond = RowVector::<u8, 3>::new(&[1, 0, 1], &device)?;
            let on_true = RowVector::<u8, 3>::new(&[1, 1, 1], &device)?;
            let on_false = RowVector::<u8, 3>::new(&[2, 2, 2], &device)?;
            let result = cond.where_cond(&on_true, &on_false)?;
            assert_eq!(result.read()?, vec![1, 2, 1]);
            Ok(())
        }
        #[test]
        fn neg() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let neg = tensor.neg()?;
            assert_eq!(neg.read()?, vec![-1.0, -2.0, -3.0]);
            Ok(())
        }

        #[test]
        fn abs() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[-1.0, 2.0, -3.0], &device)?;
            let abs = tensor.abs()?;
            assert_eq!(abs.read()?, vec![1.0, 2.0, 3.0]);
            Ok(())
        }

        #[test]
        fn tanh() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[0.0, 1.0, 2.0], &device)?;
            let tanh = tensor.tanh()?;
            let tanh_vec = tanh.read()?;
            assert_relative_eq!(tanh_vec[0], 0.0);
            assert_relative_eq!(tanh_vec[1], 0.7615942);
            assert_relative_eq!(tanh_vec[2], 0.9640276);
            Ok(())
        }

        #[test]
        fn powf() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let powered = tensor.powf(2.0)?;
            assert_eq!(powered.read()?, vec![1.0, 4.0, 9.0]);
            Ok(())
        }

        #[test]
        fn element_wise_pow() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let tensor2 = RowVector::<f32, 3>::new(&[2.0, 2.0, 2.0], &device)?;
            let powered = tensor1.pow(&tensor2)?;
            assert_eq!(powered.read()?, vec![1.0, 4.0, 9.0]);
            Ok(())
        }

        #[test]
        fn sinh() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[0.0, 1.0, -1.0], &device)?;
            let sinh = tensor.sinh()?;
            let sinh_vec = sinh.read()?;
            assert_relative_eq!(sinh_vec[0], 0.0);
            assert_relative_eq!(sinh_vec[1], 1.1752012);
            assert_relative_eq!(sinh_vec[2], -1.1752012);
            Ok(())
        }

        #[test]
        fn cosh() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[0.0, 1.0, -1.0], &device)?;
            let cosh = tensor.cosh()?;
            let cosh_vec = cosh.read()?;
            assert_relative_eq!(cosh_vec[0], 1.0);
            assert_relative_eq!(cosh_vec[1], 1.5430806);
            assert_relative_eq!(cosh_vec[2], 1.5430806);
            Ok(())
        }

        #[test]
        fn transpose() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let transposed = tensor.transpose()?;
            let trans_vec = transposed.read()?;
            assert_eq!(trans_vec, vec![1.0, 2.0, 3.0]);
            Ok(())
        }

        #[test]
        #[should_panic]
        fn invalid_size() {
            let device = Device::Cpu;
            let data: Vec<f32> = vec![1.0, 2.0]; // Wrong size (2 instead of 3)
            let _tensor = RowVector::<f32, 3>::new(&data, &device).unwrap();
        }

        #[test]
        fn multiple_operations_chain() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let result = tensor.mul_scalar(2.0)?.add_scalar(1.0)?.pow_scalar(2.0)?;
            assert_relative_eq_vec(result.read()?, vec![9.0, 25.0, 49.0]);
            Ok(())
        }

        #[test]
        fn different_dtypes() -> Result<()> {
            let device = Device::Cpu;

            // Test with f64
            let tensor_f64 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            assert_eq!(tensor_f64.0.dtype(), DType::F64);

            // Test with f32
            let tensor_f32 = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            assert_eq!(tensor_f32.0.dtype(), DType::F32);

            Ok(())
        }

        #[test]
        fn edge_cases() -> Result<()> {
            let device = Device::Cpu;

            // Test with very large numbers
            let large = RowVector::<f32, 3>::new(&[1e38, 1e38, 1e38], &device)?;
            let large_mul = large.mul_scalar(2.0)?;
            assert_relative_eq!(
                large_mul.read()?[0],
                200000000000000000000000000000000000000f32
            );

            // Test with very small numbers
            let small = RowVector::<f32, 3>::new(&[1e-38, 1e-38, 1e-38], &device)?;
            let small_div = small.div_scalar(2.0)?;
            assert!(small_div.read()?[0] != 0.0);

            // Test with zero division
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
            let zero = RowVector::<f32, 3>::zeros(&device)?;
            let div_zero = tensor.div(&zero)?;
            assert!(div_zero.read()?[0].is_infinite());

            Ok(())
        }

        #[test]
        fn broadcasting_behavior() -> Result<()> {
            let device = Device::Cpu;
            let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;

            // Test broadcasting with scalar operations
            let scalar_add = tensor.add_scalar(1.0)?;
            let scalar_mul = tensor.mul_scalar(2.0)?;

            assert_eq!(scalar_add.0.shape().dims(), [1, 3]);
            assert_eq!(scalar_mul.0.shape().dims(), [1, 3]);
            assert_eq!(scalar_add.read()?, vec![2.0, 3.0, 4.0]);
            assert_eq!(scalar_mul.read()?, vec![2.0, 4.0, 6.0]);

            Ok(())
        }
    }
}
mod column_vector {
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
            let one = Self::ones(self.0.device())?;
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
            let result = both_zero
                .where_cond(&zero, &x_zero.where_cond(&x_zero_result, &adjusted_result)?)?;

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
        fn where_cond(
            &self,
            on_true: &Self::Output,
            on_false: &Self::Output,
        ) -> Result<Self::Output> {
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
}
mod matrix {
    use crate::*;
    use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
    use std::{f64::consts::PI, marker::PhantomData};

    #[derive(Debug, Clone)]
    pub struct Matrix<T: WithDType, const ROWS: usize, const C: usize>(
        pub(crate) Tensor,
        pub(crate) PhantomData<T>,
    );
    impl<T: WithDType, const ROWS: usize, const C: usize> IsMatrix<ROWS, C> for Matrix<T, ROWS, C> {}

    impl<T: WithDType, const ROWS: usize, const COLS: usize> TensorBase<T> for Matrix<T, ROWS, COLS> {
        type ReadOutput = Vec<Vec<T>>;
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
            (ROWS, COLS)
        }
        #[inline]
        fn read(&self) -> Result<Self::ReadOutput> {
            self.0.to_vec2()
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
        pub fn new(data: &[T], device: &Device) -> Result<Matrix<T, ROWS, COLS>> {
            assert!(data.len() == ROWS * COLS);
            Ok(Self(
                Tensor::from_slice(data, (ROWS, COLS), device)?,
                PhantomData,
            ))
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ElementWiseOp<T>
        for Matrix<T, ROWS, COLS>
    {
        type Output = Matrix<T, ROWS, COLS>;
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
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ScalarOp<T> for Matrix<T, ROWS, COLS> {
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
        fn pow_scalar(&self, scalar: T) -> Result<Self> {
            let scalar_tensor = Tensor::from_vec(vec![scalar], (1, 1), self.0.device())?;
            Ok(Self(self.0.broadcast_pow(&scalar_tensor)?, PhantomData))
        }
        #[inline]
        fn powf(&self, exponent: f64) -> Result<Self> {
            Ok(Matrix(self.0.powf(exponent)?, PhantomData))
        }
        #[inline]
        fn pow(&self, other: &Self) -> Result<Self> {
            Ok(Matrix(self.0.pow(&other.0)?, PhantomData))
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> TrigOp<T> for Matrix<T, ROWS, COLS> {
        type Output = Matrix<T, ROWS, COLS>;
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
            let one = Self::ones(self.0.device())?;
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
            let ratio = self.div(x)?;
            let theta = ratio.atan()?;
            let x_lt_zero = x.lt(&zero)?;
            let y_gte_zero = self.gte(&zero)?;
            let pi = Self::ones(self.0.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_neg = Self::ones(self.0.device())?.mul_scalar(T::from_f64(-PI))?;
            let adjustment = x_lt_zero.where_cond(&y_gte_zero.where_cond(&pi, &pi_neg)?, &zero)?;
            theta.add(&adjustment)
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> UnaryOp<T> for Matrix<T, ROWS, COLS> {
        type TransposeOutput = Matrix<T, COLS, ROWS>;
        type ScalarOutput = Scalar<T>;
        #[inline]
        fn neg(&self) -> Result<Self> {
            Ok(Matrix(self.0.neg()?, PhantomData))
        }
        #[inline]
        fn abs(&self) -> Result<Self> {
            Ok(Matrix(self.0.abs()?, PhantomData))
        }
        #[inline]
        fn exp(&self) -> Result<Self> {
            Ok(Matrix(self.0.exp()?, PhantomData))
        }
        #[inline]
        fn log(&self) -> Result<Self> {
            Ok(Matrix(self.0.log()?, PhantomData))
        }
        #[inline]
        fn mean(&self) -> Result<Self::ScalarOutput> {
            Ok(Scalar(self.0.mean_all()?, PhantomData))
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> MatrixOps<T, ROWS, COLS>
        for Matrix<T, ROWS, COLS>
    {
        type MatMulMatrix<const M: usize> = Matrix<T, COLS, M>;
        type MatMulOutput<const M: usize> = Matrix<T, ROWS, M>;
        type RowSumOutput = ColumnVector<T, ROWS>;
        type ColSumOutput = RowVector<T, COLS>;
        type TransposeOutput = Matrix<T, COLS, ROWS>;
        #[inline]
        fn sum_rows(&self) -> Result<Self::RowSumOutput> {
            Ok(ColumnVector(
                self.0.sum(1)?.reshape((ROWS, 1))?,
                PhantomData,
            ))
        }
        #[inline]
        fn sum_cols(&self) -> Result<Self::ColSumOutput> {
            Ok(RowVector(self.0.sum(0)?.reshape((1, COLS))?, PhantomData))
        }
        #[inline]
        fn matmul<const O: usize>(
            &self,
            other: &Self::MatMulMatrix<O>,
        ) -> Result<Self::MatMulOutput<O>> {
            Ok(Matrix(self.0.matmul(&other.0)?, PhantomData))
        }
        #[inline]
        fn transpose(&self) -> Result<Self::TransposeOutput> {
            Ok(Matrix(self.0.t()?, PhantomData))
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ComparisonOp<T> for Matrix<T, ROWS, COLS> {
        type Output = Matrix<u8, ROWS, COLS>;
        #[inline]
        fn lt(&self, other: &Self) -> Result<Self::Output> {
            Ok(Matrix(self.0.lt(&other.0)?, PhantomData))
        }
        #[inline]
        fn lte(&self, other: &Self) -> Result<Self::Output> {
            Ok(Matrix(self.0.le(&other.0)?, PhantomData))
        }
        #[inline]
        fn eq(&self, other: &Self) -> Result<Self::Output> {
            Ok(Matrix(self.0.eq(&other.0)?, PhantomData))
        }
        #[inline]
        fn ne(&self, other: &Self) -> Result<Self::Output> {
            Ok(Matrix(self.0.ne(&other.0)?, PhantomData))
        }
        #[inline]
        fn gt(&self, other: &Self) -> Result<Self::Output> {
            Ok(Matrix(self.0.gt(&other.0)?, PhantomData))
        }
        #[inline]
        fn gte(&self, other: &Self) -> Result<Self::Output> {
            Ok(Matrix(self.0.ge(&other.0)?, PhantomData))
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> TensorFactory<T>
        for Matrix<T, ROWS, COLS>
    {
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
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ConditionalOp<T>
        for Matrix<u8, ROWS, COLS>
    {
        type Output = Matrix<T, ROWS, COLS>;
        type ComplexOutput = ComplexMatrix<T, ROWS, COLS>;
        #[inline]
        fn where_cond(
            &self,
            on_true: &Self::Output,
            on_false: &Self::Output,
        ) -> Result<Self::Output> {
            Ok(Matrix::<T, ROWS, COLS>(
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
            Ok(ComplexMatrix::<T, ROWS, COLS> {
                real: self.where_cond(&on_true.real()?, &on_false.real()?)?,
                imag: self.where_cond(&on_true.imaginary()?, &on_false.imaginary()?)?,
            })
        }
        #[inline]
        fn promote(&self, dtype: DType) -> Result<Self::Output> {
            Ok(Matrix(self.0.to_dtype(dtype)?, PhantomData))
        }
    }

    impl<const ROWS: usize, const COLS: usize> BooleanOp for Matrix<u8, ROWS, COLS> {
        type Output = Matrix<u8, ROWS, COLS>;
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
    impl<T: WithDType, const ROWS: usize, const COLS: usize> MatrixFactory<T, ROWS, COLS>
        for Matrix<T, ROWS, COLS>
    {
        #[inline]
        fn eye(device: &Device) -> Result<Self> {
            let mut data = vec![T::zero(); ROWS * COLS];
            for i in 0..ROWS.min(COLS) {
                data[i * COLS + i] = T::one();
            }
            Ok(Self(
                Tensor::from_vec(data, (ROWS, COLS), device)?,
                PhantomData,
            ))
        }
    }
    impl<F: FloatDType, const ROWS: usize, const COLS: usize> TensorFactoryFloat<F>
        for Matrix<F, ROWS, COLS>
    {
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

    #[cfg(test)]
    mod test {
        use crate::*;
        use approx::assert_relative_eq;
        use candle_core::{Device, Result};
        use operands::test::assert_relative_eq_vec_vec;

        #[test]
        fn new_tensor2d() -> Result<()> {
            let device = Device::Cpu;
            let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let tensor = Matrix::<f64, 2, 3>::new(&data, &device)?;
            assert_eq!(tensor.0.shape().dims(), [2, 3]);
            let values = tensor.read()?;
            assert_eq!(values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
            Ok(())
        }

        #[test]
        fn zeros() -> Result<()> {
            let device = Device::Cpu;
            let zeros = Matrix::<f64, 2, 3>::zeros(&device)?;
            let values = zeros.read()?;
            assert_eq!(values, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
            Ok(())
        }

        #[test]
        fn ones() -> Result<()> {
            let device = Device::Cpu;
            let ones = Matrix::<f64, 2, 3>::ones(&device)?;
            let values = ones.read()?;
            assert_eq!(values, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
            Ok(())
        }

        #[test]
        fn ones_neg() -> Result<()> {
            let device = Device::Cpu;
            let ones_neg = Matrix::<f64, 2, 3>::ones_neg(&device)?;
            let values = ones_neg.read()?;
            assert_eq!(values, vec![vec![-1.0, -1.0, -1.0], vec![-1.0, -1.0, -1.0]]);
            Ok(())
        }

        #[test]
        fn neg() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let neg = tensor.neg()?;
            let values = neg.read()?;
            assert_eq!(values, vec![vec![-1.0, -2.0], vec![-3.0, -4.0]]);
            Ok(())
        }

        #[test]
        fn abs() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[-1.0, -2.0, 3.0, -4.0], &device)?;
            let abs = tensor.abs()?;
            let values = abs.read()?;
            assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            Ok(())
        }
        #[test]
        fn add_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let result = tensor.add_scalar(2.0)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
            Ok(())
        }

        #[test]
        fn sub_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let result = tensor.sub_scalar(1.0)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
            Ok(())
        }

        #[test]
        fn mul_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let result = tensor.mul_scalar(2.0)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
            Ok(())
        }

        #[test]
        fn div_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[2.0, 4.0, 6.0, 8.0], &device)?;
            let result = tensor.div_scalar(2.0)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            Ok(())
        }

        #[test]
        fn pow_scalar() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f32, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let result = tensor.pow_scalar(2.0)?;
            let values = result.read()?;
            assert_relative_eq_vec_vec(values, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
            Ok(())
        }

        // Part 3: Element-wise Operations Tests

        #[test]
        fn add() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let tensor2 = Matrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &device)?;
            let result = tensor1.add(&tensor2)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
            Ok(())
        }

        #[test]
        fn sub() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = Matrix::<f64, 2, 2>::new(&[2.0, 3.0, 4.0, 5.0], &device)?;
            let tensor2 = Matrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &device)?;
            let result = tensor1.sub(&tensor2)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            Ok(())
        }

        #[test]
        fn mul() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let tensor2 = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
            let result = tensor1.mul(&tensor2)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
            Ok(())
        }

        #[test]
        fn div() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = Matrix::<f64, 2, 2>::new(&[2.0, 4.0, 6.0, 8.0], &device)?;
            let tensor2 = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
            let result = tensor1.div(&tensor2)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            Ok(())
        }

        #[test]
        fn element_wise_pow() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let tensor2 = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
            let result = tensor1.pow(&tensor2)?;
            let values = result.read()?;
            assert_relative_eq_vec_vec(values, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
            Ok(())
        }
        #[test]
        fn matmul() -> Result<()> {
            let device = Device::Cpu;
            let tensor1 = Matrix::<f64, 2, 3>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?;
            let tensor2 = Matrix::<f64, 3, 2>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?;
            let result = tensor1.matmul(&tensor2)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![22.0, 28.0], vec![49.0, 64.0]]);
            Ok(())
        }

        #[test]
        fn transpose() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 3>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?;
            let transposed = tensor.transpose()?;
            let values = transposed.read()?;
            assert_eq!(values, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
            Ok(())
        }

        // Part 5: Transcendental Functions Tests

        #[test]
        fn exp() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, 0.5, 2.0], &device)?;
            let result = tensor.exp()?;
            let values = result.read()?;
            assert_relative_eq!(values[0][0], 1.0);
            assert_relative_eq!(values[0][1], 2.718281828459045);
            assert_relative_eq!(values[1][0], 1.6487212707001282);
            assert_relative_eq!(values[1][1], 7.38905609893065);
            Ok(())
        }

        #[test]
        fn log() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 4.0, 8.0], &device)?;
            let result = tensor.log()?;
            let values = result.read()?;
            assert_relative_eq!(values[0][0], 0.0);
            assert_relative_eq!(values[0][1], 0.6931471805599453);
            assert_relative_eq!(values[1][0], 1.3862943611198906);
            assert_relative_eq!(values[1][1], 2.0794415416798357);
            Ok(())
        }

        #[test]
        fn sin() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(
                &[
                    0.0,
                    std::f64::consts::PI / 2.0,
                    std::f64::consts::PI,
                    3.0 * std::f64::consts::PI / 2.0,
                ],
                &device,
            )?;
            let result = tensor.sin()?;
            let values = result.read()?;
            assert_relative_eq!(values[0][0], 0.0);
            assert_relative_eq!(values[0][1], 1.0);
            assert_relative_eq!(values[1][0], 0.0);
            assert_relative_eq!(values[1][1], -1.0);
            Ok(())
        }

        #[test]
        fn cos() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(
                &[
                    0.0,
                    std::f64::consts::PI / 2.0,
                    std::f64::consts::PI,
                    3.0 * std::f64::consts::PI / 2.0,
                ],
                &device,
            )?;
            let result = tensor.cos()?;
            let values = result.read()?;
            assert_relative_eq!(values[0][0], 1.0);
            assert_relative_eq!(values[0][1], 0.0);
            assert_relative_eq!(values[1][0], -1.0);
            assert_relative_eq!(values[1][1], 0.0);
            Ok(())
        }

        #[test]
        fn sinh() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, -1.0, 2.0], &device)?;
            let result = tensor.sinh()?;
            let values = result.read()?;
            assert_relative_eq!(values[0][0], 0.0);
            assert_relative_eq!(values[0][1], 1.1752011936438014);
            assert_relative_eq!(values[1][0], -1.1752011936438014);
            assert_relative_eq!(values[1][1], 3.626860407847019);
            Ok(())
        }

        #[test]
        fn cosh() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, -1.0, 2.0], &device)?;
            let result = tensor.cosh()?;
            let values = result.read()?;
            assert_relative_eq!(values[0][0], 1.0);
            assert_relative_eq!(values[0][1], 1.5430806348152437);
            assert_relative_eq!(values[1][0], 1.5430806348152437);
            assert_relative_eq!(values[1][1], 3.7621956910836314);
            Ok(())
        }

        #[test]
        fn tanh() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, -1.0, 2.0], &device)?;
            let result = tensor.tanh()?;
            let values = result.read()?;
            assert_relative_eq!(values[0][0], 0.0);
            assert_relative_eq!(values[0][1], 0.7615941559557649);
            assert_relative_eq!(values[1][0], -0.7615941559557649);
            assert_relative_eq!(values[1][1], 0.9640275800758169);
            Ok(())
        }

        #[test]
        fn powf() -> Result<()> {
            let device = Device::Cpu;
            let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
            let result = tensor.powf(2.0)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
            Ok(())
        }

        #[test]
        fn where_cond() -> Result<()> {
            let device = Device::Cpu;
            let condition = Matrix::<u8, 2, 2>::new(&[1, 0, 1, 0], &device)?;
            let on_true = Matrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &device)?;
            let on_false = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
            let result = condition.where_cond(&on_true, &on_false)?;
            let values = result.read()?;
            assert_eq!(values, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
            Ok(())
        }
    }
}
mod complex_scalar {
    use crate::ops::*;
    use crate::Scalar;
    use candle_core::{DType, Device, FloatDType, Result, WithDType};
    use std::{f64::consts::PI, marker::PhantomData};
    #[derive(Debug, Clone)]
    pub struct ComplexScalar<T: WithDType> {
        pub(crate) real: Scalar<T>,
        pub(crate) imag: Scalar<T>,
    }
    impl<T: WithDType> IsScalar for ComplexScalar<T> {}
    impl<T: WithDType> TensorBase<T> for ComplexScalar<T> {
        type ReadOutput = (T, T);
        #[inline]
        fn device(&self) -> &Device {
            self.real.device()
        }
        #[inline]
        fn dtype() -> DType {
            T::DTYPE
        }
        #[inline]
        fn shape() -> (usize, usize) {
            (1, 1)
        }
        #[inline]
        fn read(&self) -> Result<Self::ReadOutput> {
            Ok((self.real.read()?, self.imag.read()?))
        }
    }
    impl<T: WithDType> ComplexScalar<T> {
        pub fn new(real: T, imag: T, device: &Device) -> Result<ComplexScalar<T>> {
            Ok(Self {
                real: Scalar::new(real, device)?,
                imag: Scalar::new(imag, device)?,
            })
        }
    }
    impl<T: WithDType> ElementWiseOp<T> for ComplexScalar<T> {
        type Output = ComplexScalar<T>;
        #[inline]
        fn add(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.add(&rhs.real)?,
                imag: self.imag.add(&rhs.imag)?,
            })
        }
        #[inline]
        fn sub(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sub(&rhs.real)?,
                imag: self.imag.sub(&rhs.imag)?,
            })
        }
        #[inline]
        fn mul(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.mul(&rhs.real)?.sub(&self.imag.mul(&rhs.imag)?)?,
                imag: self.real.mul(&rhs.imag)?.add(&self.imag.mul(&rhs.real)?)?,
            })
        }
        #[inline]
        fn div(&self, rhs: &Self) -> Result<Self::Output> {
            let denom = rhs.real.mul(&rhs.real)?.add(&rhs.imag.mul(&rhs.imag)?)?;
            Ok(Self {
                real: self
                    .real
                    .mul(&rhs.real)?
                    .add(&self.imag.mul(&rhs.imag)?)?
                    .div(&denom)?,
                imag: self
                    .imag
                    .mul(&rhs.real)?
                    .sub(&self.real.mul(&rhs.imag)?)?
                    .div(&denom)?,
            })
        }
        #[inline]
        fn clamp(&self, min: &T, max: &T) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clamp(min, max)?,
                imag: self.imag.clamp(min, max)?,
            })
        }
    }
    impl<T: WithDType> ScalarOp<T> for ComplexScalar<T> {
        #[inline]
        fn add_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.add_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn sub_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.sub_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn mul_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.mul_scalar(scalar)?,
                imag: self.imag.mul_scalar(scalar)?,
            })
        }
        #[inline]
        fn div_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.div_scalar(scalar)?,
                imag: self.imag.div_scalar(scalar)?,
            })
        }
        #[inline]
        fn powf(&self, exponent: f64) -> Result<Self> {
            let r = self.magnitude()?;
            let theta = self.arg()?;
            let r_pow = r.powf(exponent)?;
            let new_theta = theta.mul_scalar(T::from_f64(exponent))?;
            Ok(Self {
                real: r_pow.mul(&new_theta.cos()?)?,
                imag: r_pow.mul(&new_theta.sin()?)?,
            })
        }
        #[inline]
        fn pow(&self, other: &Self) -> Result<Self> {
            self.log()?.mul(other)?.exp()
        }
        #[inline]
        fn pow_scalar(&self, scalar: T) -> Result<Self> {
            let r = self.magnitude()?;
            let theta = self.arg()?;
            let r_pow = r.pow_scalar(scalar)?;
            let new_theta = theta.mul_scalar(scalar)?;
            Ok(Self {
                real: r_pow.mul(&new_theta.cos()?)?,
                imag: r_pow.mul(&new_theta.sin()?)?,
            })
        }
    }
    impl<T: WithDType> TrigOp<T> for ComplexScalar<T> {
        type Output = ComplexScalar<T>;
        #[inline]
        fn sin(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sin()?.mul(&self.imag.cosh()?)?,
                imag: self.real.cos()?.mul(&self.imag.sinh()?)?,
            })
        }
        #[inline]
        fn cos(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cos()?.mul(&self.imag.cosh()?)?,
                imag: self.real.sin()?.mul(&self.imag.sinh()?)?.neg()?,
            })
        }
        #[inline]
        fn sinh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sinh()?.mul(&self.imag.cos()?)?,
                imag: self.real.cosh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn cosh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cosh()?.mul(&self.imag.cos()?)?,
                imag: self.real.sinh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn tanh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.tanh()?,
                imag: self.imag.tanh()?,
            })
        }
        #[inline]
        fn atan(&self) -> Result<Self::Output> {
            let zero = Scalar::zeros(self.device())?;
            let one = Scalar::ones(self.device())?;
            let eps = Scalar::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi_half = Scalar::ones(self.device())?.mul_scalar(T::from_f64(PI / 2.0))?;

            // Form iz
            let iz = ComplexScalar {
                real: self.imag.neg()?,
                imag: self.real.clone(),
            };

            // Compute (1 + iz)/(1 - iz)
            let numerator = ComplexScalar {
                real: one.clone(),
                imag: zero.clone(),
            }
            .add(&iz)?;

            let denominator = ComplexScalar {
                real: one.clone(),
                imag: zero.clone(),
            }
            .sub(&iz)?;

            // Check for special cases
            let mag_real = self.real.abs()?;
            let mag_imag = self.imag.abs()?;

            // z = 0 case
            let is_zero = mag_real.lt(&eps)?.mul(&mag_imag.lt(&eps)?)?;

            // z = ±i case (near branch points)
            let near_i = mag_real
                .lt(&eps)?
                .mul(&(mag_imag.sub(&one)?.abs()?.lt(&eps)?))?;

            // Standard computation
            let standard_result = {
                let ratio = numerator.div(&denominator)?;
                let log_result = ratio.log()?;
                ComplexScalar {
                    real: log_result.imag.mul_scalar(T::from_f64(0.5))?,
                    imag: log_result.real.mul_scalar(T::from_f64(-0.5))?,
                }
            };

            // Special case results
            let zero_result = ComplexScalar {
                real: zero.clone(),
                imag: zero.clone(),
            };

            let i_result = ComplexScalar {
                real: pi_half.mul(&self.imag.div(&mag_imag)?)?,
                imag: Scalar::ones(self.device())?.mul_scalar(T::from_f64(f64::INFINITY))?,
            };

            // Combine results using conditional operations
            let result = is_zero.where_cond_complex(
                &zero_result,
                &near_i.where_cond_complex(&i_result, &standard_result)?,
            )?;

            Ok(result)
        }
        #[inline]
        fn atan2(&self, x: &Self) -> Result<Self::Output> {
            let zero = Scalar::zeros(self.device())?;
            let eps = Scalar::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi = Scalar::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

            // Compute magnitudes
            let y_mag = self.magnitude()?;
            let x_mag = x.magnitude()?;

            // Special cases
            let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
            let x_zero = x_mag.lt(&eps)?;

            // Base atan computation
            let z = self.div(x)?;
            let base_atan = z.atan()?;

            // Quadrant adjustments
            let x_neg = x.real.lt(&zero)?;
            let y_gte_zero = self.real.gte(&zero)?;

            // When x < 0: add π for y ≥ 0, subtract π for y < 0
            let adjustment = x_neg.where_cond(&y_gte_zero.where_cond(&pi, &pi.neg()?)?, &zero)?;

            // Apply adjustment to real part only
            let adjusted_result = ComplexScalar {
                real: base_atan.real.add(&adjustment)?,
                imag: base_atan.imag,
            };

            // Handle x = 0 cases
            let x_zero_result = ComplexScalar {
                real: y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?,
                imag: zero.clone(),
            };

            // Combine all cases
            let result = both_zero.where_cond_complex(
                &ComplexScalar {
                    real: zero.clone(),
                    imag: zero.clone(),
                },
                &x_zero.where_cond_complex(&x_zero_result, &adjusted_result)?,
            )?;

            Ok(result)
        }
    }
    impl<T: WithDType> UnaryOp<T> for ComplexScalar<T> {
        type TransposeOutput = ComplexScalar<T>;
        type ScalarOutput = ComplexScalar<T>;
        #[inline]
        fn neg(&self) -> Result<Self> {
            Ok(ComplexScalar {
                real: self.real.neg()?,
                imag: self.imag.neg()?,
            })
        }
        #[inline]
        fn abs(&self) -> Result<Self> {
            Ok(Self {
                real: self.real.abs()?,
                imag: Scalar::zeros(self.real.device())?,
            })
        }
        #[inline]
        fn exp(&self) -> Result<Self> {
            let exp_real = self.real.exp()?;
            Ok(Self {
                real: exp_real.mul(&self.imag.cos()?)?,
                imag: exp_real.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn log(&self) -> Result<Self> {
            Ok(Self {
                real: self.magnitude()?.log()?,
                imag: self.arg()?,
            })
        }
        #[inline]
        fn mean(&self) -> Result<Self::ScalarOutput> {
            Ok(self.clone())
        }
    }
    impl<T: WithDType> ComparisonOp<T> for ComplexScalar<T> {
        type Output = Scalar<u8>;
        #[inline]
        fn lt(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.lt(&mag_other)
        }
        #[inline]
        fn lte(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.lte(&mag_other)
        }
        #[inline]
        fn eq(&self, other: &Self) -> Result<Self::Output> {
            let real_eq = self.real.eq(&other.real)?;
            let imag_eq = self.imag.eq(&other.imag)?;
            real_eq.mul(&imag_eq)
        }
        #[inline]
        fn ne(&self, other: &Self) -> Result<Self::Output> {
            let real_ne = self.real.ne(&other.real)?;
            let imag_ne = self.imag.ne(&other.imag)?;
            real_ne
                .add(&imag_ne)?
                .gt(&Scalar::<u8>::zeros(self.device())?)
        }
        #[inline]
        fn gt(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.gt(&mag_other)
        }
        #[inline]
        fn gte(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.gte(&mag_other)
        }
    }
    impl<T: WithDType> ComplexOp<T> for ComplexScalar<T> {
        type Output = ComplexScalar<T>;
        type RealOutput = Scalar<T>;
        #[inline]
        fn conj(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clone(),
                imag: self.imag.neg()?,
            })
        }
        #[inline]
        fn magnitude(&self) -> Result<Self::RealOutput> {
            Ok(Scalar(
                self.real.0.sqr()?.add(&self.imag.0.sqr()?)?.sqrt()?,
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
    impl<T: WithDType> TensorFactory<T> for ComplexScalar<T> {
        #[inline]
        fn zeros(device: &Device) -> Result<Self> {
            Ok(Self {
                real: Scalar::zeros(device)?,
                imag: Scalar::zeros(device)?,
            })
        }
        #[inline]
        fn ones(device: &Device) -> Result<Self> {
            Ok(Self {
                real: Scalar::ones(device)?,
                imag: Scalar::zeros(device)?,
            })
        }
        #[inline]
        fn ones_neg(device: &Device) -> Result<Self> {
            Ok(Self {
                real: Scalar::ones_neg(device)?,
                imag: Scalar::zeros(device)?,
            })
        }
    }
    impl<F: FloatDType> TensorFactoryFloat<F> for ComplexScalar<F> {
        #[inline]
        fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: Scalar::<F>::randn(mean, std, device)?,
                imag: Scalar::<F>::randn(mean, std, device)?,
            })
        }
        #[inline]
        fn randu(low: F, high: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: Scalar::<F>::randu(low, high, device)?,
                imag: Scalar::<F>::randu(low, high, device)?,
            })
        }
    }
    impl<T: WithDType> RealComplexOp<T> for ComplexScalar<T> {
        type Output = ComplexScalar<T>;
        type Input = Scalar<T>;
        #[inline]
        fn mul_complex(&self, rhs: &Self::Input) -> Result<Self::Output> {
            Ok(ComplexScalar {
                real: self.real.mul(rhs)?,
                imag: self.imag.mul(rhs)?,
            })
        }
    }
}
mod complex_row_vector {
    use crate::ops::*;
    use crate::{ComplexColumnVector, ComplexMatrix, ComplexScalar, RowVector, Scalar};
    use candle_core::{DType, Device, FloatDType, Result, WithDType};
    use std::{f64::consts::PI, marker::PhantomData};
    #[derive(Debug, Clone)]
    pub struct ComplexRowVector<T: WithDType, const ROWS: usize> {
        pub(crate) real: RowVector<T, ROWS>,
        pub(crate) imag: RowVector<T, ROWS>,
    }
    impl<T: WithDType, const ROWS: usize> IsRowVector<ROWS> for ComplexRowVector<T, ROWS> {}
    impl<T: WithDType, const COLS: usize> TensorBase<T> for ComplexRowVector<T, COLS> {
        type ReadOutput = (Vec<T>, Vec<T>);
        #[inline]
        fn device(&self) -> &Device {
            self.real.device()
        }
        #[inline]
        fn dtype() -> DType {
            T::DTYPE
        }
        #[inline]
        fn shape() -> (usize, usize) {
            (1, COLS)
        }
        #[inline]
        fn read(&self) -> Result<Self::ReadOutput> {
            Ok((self.real.read()?, self.imag.read()?))
        }
    }
    impl<T: WithDType, const COLS: usize> ComplexRowVector<T, COLS> {
        pub fn new(real: &[T], imag: &[T], device: &Device) -> Result<ComplexRowVector<T, COLS>> {
            assert!(real.len() == COLS);
            assert!(imag.len() == COLS);
            Ok(Self {
                real: RowVector::<T, COLS>::new(real, device)?,
                imag: RowVector::<T, COLS>::new(imag, device)?,
            })
        }
    }
    impl<T: WithDType, const COLS: usize> ElementWiseOp<T> for ComplexRowVector<T, COLS> {
        type Output = ComplexRowVector<T, COLS>;
        #[inline]
        fn add(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.add(&rhs.real)?,
                imag: self.imag.add(&rhs.imag)?,
            })
        }
        #[inline]
        fn sub(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sub(&rhs.real)?,
                imag: self.imag.sub(&rhs.imag)?,
            })
        }
        #[inline]
        fn mul(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.mul(&rhs.real)?.sub(&self.imag.mul(&rhs.imag)?)?,
                imag: self.real.mul(&rhs.imag)?.add(&self.imag.mul(&rhs.real)?)?,
            })
        }
        #[inline]
        fn div(&self, rhs: &Self) -> Result<Self::Output> {
            let denom = rhs.real.mul(&rhs.real)?.add(&rhs.imag.mul(&rhs.imag)?)?;
            Ok(Self {
                real: self
                    .real
                    .mul(&rhs.real)?
                    .add(&self.imag.mul(&rhs.imag)?)?
                    .div(&denom)?,
                imag: self
                    .imag
                    .mul(&rhs.real)?
                    .sub(&self.real.mul(&rhs.imag)?)?
                    .div(&denom)?,
            })
        }
        #[inline]
        fn clamp(&self, min: &T, max: &T) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clamp(min, max)?,
                imag: self.imag.clamp(min, max)?,
            })
        }
    }
    impl<T: WithDType, const COLS: usize> ScalarOp<T> for ComplexRowVector<T, COLS> {
        #[inline]
        fn add_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.add_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn sub_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.sub_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn mul_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.mul_scalar(scalar)?,
                imag: self.imag.mul_scalar(scalar)?,
            })
        }
        #[inline]
        fn div_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.div_scalar(scalar)?,
                imag: self.imag.div_scalar(scalar)?,
            })
        }
        #[inline]
        fn pow_scalar(&self, scalar: T) -> Result<Self> {
            let r = self.magnitude()?;
            let theta = self.imag.atan2(&self.real)?; // Use atan2 for correct argument
            let r_pow = r.pow_scalar(scalar)?;
            let new_theta = theta.mul_scalar(scalar)?;
            Ok(Self {
                real: r_pow.mul(&new_theta.cos()?)?,
                imag: r_pow.mul(&new_theta.sin()?)?,
            })
        }
        #[inline]
        fn powf(&self, exponent: f64) -> Result<Self> {
            let r = self.magnitude()?;
            let theta = self.imag.atan2(&self.real)?;
            let r_pow = r.powf(exponent)?;
            let new_theta = theta.mul_scalar(T::from_f64(exponent))?;
            Ok(Self {
                real: r_pow.mul(&new_theta.cos()?)?,
                imag: r_pow.mul(&new_theta.sin()?)?,
            })
        }
        #[inline]
        fn pow(&self, other: &Self) -> Result<Self> {
            self.log()?.mul(other)?.exp()
        }
    }
    impl<T: WithDType, const COLS: usize> TrigOp<T> for ComplexRowVector<T, COLS> {
        type Output = ComplexRowVector<T, COLS>;
        #[inline]
        fn sin(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sin()?.mul(&self.imag.cosh()?)?,
                imag: self.real.cos()?.mul(&self.imag.sinh()?)?,
            })
        }
        #[inline]
        fn cos(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cos()?.mul(&self.imag.cosh()?)?,
                imag: self.real.sin()?.mul(&self.imag.sinh()?)?.neg()?,
            })
        }
        #[inline]
        fn sinh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sinh()?.mul(&self.imag.cos()?)?,
                imag: self.real.cosh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn cosh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cosh()?.mul(&self.imag.cos()?)?,
                imag: self.real.sinh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn tanh(&self) -> Result<Self::Output> {
            let two_x = self.real.mul_scalar(T::from_f64(2.0))?;
            let two_y = self.imag.mul_scalar(T::from_f64(2.0))?;

            let sinh_2x = two_x.sinh()?;
            let sin_2y = two_y.sin()?;
            let cosh_2x = two_x.cosh()?;
            let cos_2y = two_y.cos()?;

            let denom = cosh_2x.add(&cos_2y)?;

            Ok(Self {
                real: sinh_2x.div(&denom)?,
                imag: sin_2y.div(&denom)?,
            })
        }
        #[inline]
        fn atan(&self) -> Result<Self::Output> {
            let zero = RowVector::zeros(self.device())?;
            let one = RowVector::ones(self.device())?;
            let eps = RowVector::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi_half = RowVector::ones(self.device())?.mul_scalar(T::from_f64(PI / 2.0))?;

            // Form iz
            let iz = ComplexRowVector {
                real: self.imag.neg()?,
                imag: self.real.clone(),
            };

            // Compute (1 + iz)/(1 - iz)
            let numerator = ComplexRowVector {
                real: one.clone(),
                imag: zero.clone(),
            }
            .add(&iz)?;

            let denominator = ComplexRowVector {
                real: one.clone(),
                imag: zero.clone(),
            }
            .sub(&iz)?;

            // Check for special cases
            let mag_real = self.real.abs()?;
            let mag_imag = self.imag.abs()?;

            // z = 0 case
            let is_zero = mag_real.lt(&eps)?.mul(&mag_imag.lt(&eps)?)?;

            // z = ±i case (near branch points)
            let near_i = mag_real
                .lt(&eps)?
                .mul(&(mag_imag.sub(&one)?.abs()?.lt(&eps)?))?;

            // Standard computation
            let standard_result = {
                let ratio = numerator.div(&denominator)?;
                let log_result = ratio.log()?;
                ComplexRowVector {
                    real: log_result.imag.mul_scalar(T::from_f64(0.5))?,
                    imag: log_result.real.mul_scalar(T::from_f64(-0.5))?,
                }
            };

            // Special case results
            let zero_result = ComplexRowVector {
                real: zero.clone(),
                imag: zero.clone(),
            };

            let i_result = ComplexRowVector {
                real: pi_half.mul(&self.imag.div(&mag_imag)?)?,
                imag: RowVector::ones(self.device())?.mul_scalar(T::from_f64(f64::INFINITY))?,
            };

            // Combine results using conditional operations
            let result = is_zero.where_cond_complex(
                &zero_result,
                &near_i.where_cond_complex(&i_result, &standard_result)?,
            )?;

            Ok(result)
        }
        #[inline]
        fn atan2(&self, x: &Self) -> Result<Self::Output> {
            let zero = RowVector::zeros(self.device())?;
            let eps = RowVector::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi = RowVector::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

            // Compute magnitudes
            let y_mag = self.magnitude()?;
            let x_mag = x.magnitude()?;

            // Special cases
            let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
            let x_zero = x_mag.lt(&eps)?;

            // Base atan computation
            let z = self.div(x)?;
            let base_atan = z.atan()?;

            // Quadrant adjustments
            let x_neg = x.real.lt(&zero)?;
            let y_gte_zero = self.real.gte(&zero)?;

            // When x < 0: add π for y ≥ 0, subtract π for y < 0
            let adjustment = x_neg.where_cond(&y_gte_zero.where_cond(&pi, &pi.neg()?)?, &zero)?;

            // Apply adjustment to real part only
            let adjusted_result = ComplexRowVector {
                real: base_atan.real.add(&adjustment)?,
                imag: base_atan.imag,
            };

            // Handle x = 0 cases
            let x_zero_result = ComplexRowVector {
                real: y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?,
                imag: zero.clone(),
            };

            // Combine all cases
            let result = both_zero.where_cond_complex(
                &ComplexRowVector {
                    real: zero.clone(),
                    imag: zero.clone(),
                },
                &x_zero.where_cond_complex(&x_zero_result, &adjusted_result)?,
            )?;

            Ok(result)
        }
    }
    impl<T: WithDType, const COLS: usize> UnaryOp<T> for ComplexRowVector<T, COLS> {
        type TransposeOutput = ComplexMatrix<T, 1, COLS>;
        type ScalarOutput = ComplexScalar<T>;
        #[inline]
        fn neg(&self) -> Result<Self> {
            Ok(Self {
                real: self.real.mul_scalar(T::from_f64(-1.0))?,
                imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
            })
        }
        #[inline]
        fn abs(&self) -> Result<Self> {
            Ok(Self {
                real: self.magnitude()?,
                imag: RowVector::zeros(self.real.device())?,
            })
        }
        #[inline]
        fn exp(&self) -> Result<Self> {
            let exp_real = self.real.exp()?;
            let cos_imag = self.imag.cos()?;
            let sin_imag = self.imag.sin()?;
            Ok(Self {
                real: exp_real.mul(&cos_imag)?,
                imag: exp_real.mul(&sin_imag)?,
            })
        }
        #[inline]
        fn log(&self) -> Result<Self> {
            let abs = self.magnitude()?;
            let zero = RowVector::<T, COLS>::zeros(self.real.0.device())?;
            let x_gt_zero = self.real.gt(&zero)?;
            let x_lt_zero = self.real.lt(&zero)?;
            let y_gte_zero = self.imag.gte(&zero)?;
            let ratio = self.imag.div(&self.real)?;
            let base_angle = ratio.tanh()?;
            let pi =
                RowVector::<T, COLS>::ones(self.real.0.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_neg = pi.neg()?;
            let adjustment = x_lt_zero.where_cond(&y_gte_zero.where_cond(&pi, &pi_neg)?, &zero)?;
            let x_is_zero = self
                .real
                .abs()?
                .lt(&RowVector::<T, COLS>::ones(self.real.0.device())?
                    .mul_scalar(T::from_f64(1e-10))?)?;
            let half_pi = pi.mul_scalar(T::from_f64(0.5))?;
            let neg_half_pi = half_pi.neg()?;
            let arg = x_is_zero.where_cond(
                &y_gte_zero.where_cond(&half_pi, &neg_half_pi)?,
                &base_angle.add(&adjustment)?,
            )?;

            Ok(Self {
                real: abs.log()?,
                imag: arg,
            })
        }
        #[inline]
        fn mean(&self) -> Result<ComplexScalar<T>> {
            Ok(ComplexScalar {
                real: self.real.mean()?,
                imag: self.imag.mean()?,
            })
        }
    }
    impl<T: WithDType, const COLS: usize> ComparisonOp<T> for ComplexRowVector<T, COLS> {
        type Output = RowVector<u8, COLS>;
        #[inline]
        fn lt(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.lt(&mag_other)
        }
        #[inline]
        fn lte(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.lte(&mag_other)
        }
        #[inline]
        fn eq(&self, other: &Self) -> Result<Self::Output> {
            let real_eq = self.real.eq(&other.real)?;
            let imag_eq = self.imag.eq(&other.imag)?;
            real_eq.mul(&imag_eq)
        }
        #[inline]
        fn ne(&self, other: &Self) -> Result<Self::Output> {
            let real_ne = self.real.ne(&other.real)?;
            let imag_ne = self.imag.ne(&other.imag)?;
            real_ne
                .add(&imag_ne)?
                .gt(&RowVector::<u8, COLS>::zeros(self.device())?)
        }
        #[inline]
        fn gt(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.gt(&mag_other)
        }
        #[inline]
        fn gte(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other: RowVector<T, COLS> = other.magnitude()?;
            mag_self.gte(&mag_other)
        }
    }
    impl<T: WithDType, const COLS: usize> RowVectorOps<T, COLS> for ComplexRowVector<T, COLS> {
        type DotInput = ComplexRowVector<T, COLS>;
        type DotOutput = ComplexScalar<T>;
        type MatMulMatrix<const M: usize> = ComplexMatrix<T, COLS, M>;
        type MatMulOutput<const M: usize> = ComplexRowVector<T, M>;
        type TransposeOutput = ComplexColumnVector<T, COLS>;
        type BroadcastOutput<const R: usize> = ComplexMatrix<T, R, COLS>;
        #[inline]
        fn dot(&self, other: &Self::DotInput) -> Result<Self::DotOutput> {
            let conj_self = self.conj()?;

            Ok(ComplexScalar {
                real: conj_self
                    .real
                    .dot(&other.real)?
                    .add(&conj_self.imag.dot(&other.imag)?)?,
                imag: Scalar::zeros(self.real.device())?,
            })
        }
        #[inline]
        fn matmul<const M: usize>(
            &self,
            other: &Self::MatMulMatrix<M>,
        ) -> Result<Self::MatMulOutput<M>> {
            Ok(ComplexRowVector {
                real: self
                    .real
                    .matmul(&other.real)?
                    .sub(&self.imag.matmul(&other.imag)?)?,
                imag: self
                    .real
                    .matmul(&other.imag)?
                    .add(&self.imag.matmul(&other.real)?)?,
            })
        }
        #[inline]
        fn broadcast<const R: usize>(&self) -> Result<Self::BroadcastOutput<R>> {
            Ok(ComplexMatrix {
                real: self.real.broadcast::<R>()?,
                imag: self.imag.broadcast::<R>()?,
            })
        }
        #[inline]
        fn transpose(&self) -> Result<Self::TransposeOutput> {
            Ok(ComplexColumnVector {
                real: self.real.transpose()?,
                imag: self.imag.transpose()?,
            })
        }
    }
    impl<T: WithDType, const COLS: usize> ComplexOp<T> for ComplexRowVector<T, COLS> {
        type Output = ComplexRowVector<T, COLS>;
        type RealOutput = RowVector<T, COLS>;
        #[inline]
        fn conj(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clone(),
                imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
            })
        }
        #[inline]
        fn magnitude(&self) -> Result<Self::RealOutput> {
            Ok(RowVector(
                self.real.0.sqr()?.add(&self.imag.0.sqr()?)?.sqrt()?,
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
    impl<T: WithDType, const COLS: usize> TensorFactory<T> for ComplexRowVector<T, COLS> {
        #[inline]
        fn zeros(device: &Device) -> Result<Self> {
            Ok(Self {
                real: RowVector::zeros(device)?,
                imag: RowVector::zeros(device)?,
            })
        }
        #[inline]
        fn ones(device: &Device) -> Result<Self> {
            Ok(Self {
                real: RowVector::ones(device)?,
                imag: RowVector::zeros(device)?,
            })
        }
        #[inline]
        fn ones_neg(device: &Device) -> Result<Self> {
            Ok(Self {
                real: RowVector::ones_neg(device)?,
                imag: RowVector::zeros(device)?,
            })
        }
    }
    impl<F: FloatDType, const COLS: usize> TensorFactoryFloat<F> for ComplexRowVector<F, COLS> {
        #[inline]
        fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: RowVector::<F, COLS>::randn(mean, std, device)?,
                imag: RowVector::<F, COLS>::randn(mean, std, device)?,
            })
        }
        #[inline]
        fn randu(low: F, high: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: RowVector::<F, COLS>::randu(low, high, device)?,
                imag: RowVector::<F, COLS>::randu(low, high, device)?,
            })
        }
    }
    impl<T: WithDType, const COLS: usize> RealComplexOp<T> for ComplexRowVector<T, COLS> {
        type Output = ComplexRowVector<T, COLS>;
        type Input = RowVector<T, COLS>;
        #[inline]
        fn mul_complex(&self, rhs: &Self::Input) -> Result<Self::Output> {
            Ok(ComplexRowVector {
                real: self.real.mul(rhs)?,
                imag: self.imag.mul(rhs)?,
            })
        }
    }
    #[cfg(test)]
    mod test {
        use crate::*;
        use approx::assert_relative_eq;
        use candle_core::{Device, Result};
        use operands::test::assert_relative_eq_vec;
        use std::f64::consts::PI;

        #[test]
        fn new_complex_tensor() -> Result<()> {
            let device = Device::Cpu;
            let real = vec![1.0, 2.0, 3.0];
            let imag = vec![4.0, 5.0, 6.0];
            let tensor = ComplexRowVector::<f64, 3>::new(&real, &imag, &device)?;
            assert_eq!(tensor.real.read()?, real);
            assert_eq!(tensor.imag.read()?, imag);
            Ok(())
        }

        #[test]
        fn zeros() -> Result<()> {
            let device = Device::Cpu;
            let zeros = ComplexRowVector::<f64, 3>::zeros(&device)?;
            assert_eq!(zeros.real.read()?, vec![0.0, 0.0, 0.0]);
            assert_eq!(zeros.imag.read()?, vec![0.0, 0.0, 0.0]);
            Ok(())
        }

        #[test]
        fn ones() -> Result<()> {
            let device = Device::Cpu;
            let ones = ComplexRowVector::<f64, 3>::ones(&device)?;
            assert_eq!(ones.real.read()?, vec![1.0, 1.0, 1.0]);
            assert_eq!(ones.imag.read()?, vec![0.0, 0.0, 0.0]);
            Ok(())
        }

        #[test]
        fn ones_neg() -> Result<()> {
            let device = Device::Cpu;
            let ones_neg = ComplexRowVector::<f64, 3>::ones_neg(&device)?;
            assert_eq!(ones_neg.real.read()?, vec![-1.0, -1.0, -1.0]);
            assert_eq!(ones_neg.imag.read()?, vec![0.0, 0.0, 0.0]);
            Ok(())
        }

        #[test]
        fn add() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;
            let b = ComplexRowVector::<f64, 2>::new(&[5.0, 6.0], &[7.0, 8.0], &device)?;
            let c = a.add(&b)?;
            assert_eq!(c.real.read()?, vec![6.0, 8.0]);
            assert_eq!(c.imag.read()?, vec![10.0, 12.0]);
            Ok(())
        }

        #[test]
        fn sub() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;
            let b = ComplexRowVector::<f64, 2>::new(&[5.0, 6.0], &[7.0, 8.0], &device)?;
            let c = a.sub(&b)?;
            assert_eq!(c.real.read()?, vec![-4.0, -4.0]);
            assert_eq!(c.imag.read()?, vec![-4.0, -4.0]);
            Ok(())
        }

        #[test]
        fn mul() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[1.0, 1.0], &device)?;
            let b = ComplexRowVector::<f64, 2>::new(&[2.0, 3.0], &[1.0, 1.0], &device)?;
            let c = a.mul(&b)?;
            // (1 + i)(2 + i) = (2 - 1) + (2 + 1)i = 1 + 3i
            // (2 + i)(3 + i) = (6 - 1) + (3 + 2)i = 5 + 5i
            assert_relative_eq_vec(c.real.read()?, vec![1.0, 5.0]);
            assert_relative_eq_vec(c.imag.read()?, vec![3.0, 5.0]);
            Ok(())
        }
        #[test]
        fn div_pure_real() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 1>::new(&[4.0], &[0.0], &device)?;
            let b = ComplexRowVector::<f64, 1>::new(&[2.0], &[0.0], &device)?;
            let c = a.div(&b)?;
            // (4 + 2i)/(2 + i) = (10 + 0i)/5 = 2 + 0i
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0], 2.0);
            assert_relative_eq!(imag[0], 0.0);
            Ok(())
        }
        #[test]
        fn div_pure_imag() -> Result<()> {
            // Case 1: i/i = 1
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[1.0], &device)?;
            let b = ComplexRowVector::<f64, 1>::new(&[0.0], &[1.0], &device)?;
            let c = a.div(&b)?;
            assert_relative_eq!(c.real.read()?[0], 1.0);
            assert_relative_eq!(c.imag.read()?[0], 0.0);
            Ok(())
        }
        #[test]
        fn div_unit_circle() -> Result<()> {
            // Case 2: (1 + i)/(1 - i) = 0 + i
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 1>::new(&[1.0], &[1.0], &device)?;
            let b = ComplexRowVector::<f64, 1>::new(&[1.0], &[-1.0], &device)?;
            let c = a.div(&b)?;
            assert_relative_eq!(c.real.read()?[0], 0.0);
            assert_relative_eq!(c.imag.read()?[0], 1.0);
            Ok(())
        }
        #[test]
        fn div_psychotic() -> Result<()> {
            // Case 3: (3 + 4i)/(2 + 2i) = 1.75 + 0.25i
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 1>::new(&[3.0], &[4.0], &device)?;
            let b = ComplexRowVector::<f64, 1>::new(&[2.0], &[2.0], &device)?;
            let c = a.div(&b)?;
            assert_relative_eq!(c.real.read()?[0], 1.75);
            assert_relative_eq!(c.imag.read()?[0], 0.25);

            Ok(())
        }

        #[test]
        fn exp() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[PI / 2.0], &device)?;
            let c = a.exp()?;
            // e^(iπ/2) = i
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0], 0.0);
            assert_relative_eq!(imag[0], 1.0);
            Ok(())
        }

        #[test]
        fn log() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[1.0], &device)?;
            let c = a.log()?;
            // ln(i) = iπ/2
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0], 0.0);
            assert_relative_eq!(imag[0], PI / 2.0);
            Ok(())
        }

        #[test]
        fn conj() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;
            let c = a.conj()?;
            assert_eq!(c.real.read()?, vec![1.0, 2.0]);
            assert_eq!(c.imag.read()?, vec![-3.0, -4.0]);
            Ok(())
        }

        #[test]
        fn abs() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 2>::new(&[3.0, 0.0], &[4.0, 1.0], &device)?;
            let c = UnaryOp::abs(&a)?;
            // |3 + 4i| = 5, |0 + i| = 1
            assert_eq!(c.real()?.read()?, vec![5.0, 1.0]); // Correct: Accessing c.real
            assert_eq!(c.imaginary()?.read()?, vec![0.0, 0.0]); // Correct: Verifying imaginary part is zero
            Ok(())
        }

        #[test]
        fn where_cond() -> Result<()> {
            let device = Device::Cpu;
            let cond = RowVector::<u8, 2>::new(&[1, 0], &device)?;
            let on_true = ComplexRowVector::<f64, 2>::new(&[1.0, 1.0], &[1.0, 1.0], &device)?;
            let on_false = ComplexRowVector::<f64, 2>::new(&[2.0, 2.0], &[2.0, 2.0], &device)?;
            let result = cond.where_cond_complex(&on_true, &on_false)?;
            assert_eq!(result.real.read()?, vec![1.0, 2.0]);
            assert_eq!(result.imag.read()?, vec![1.0, 2.0]);
            Ok(())
        }

        #[test]
        fn scalar_operations() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;

            // Test add_scalar
            let add_result = a.add_scalar(2.0)?;
            assert_eq!(add_result.real.read()?, vec![3.0, 4.0]);
            assert_eq!(add_result.imag.read()?, vec![3.0, 4.0]);

            // Test mul_scalar
            let mul_result = a.mul_scalar(2.0)?;
            assert_eq!(mul_result.real.read()?, vec![2.0, 4.0]);
            assert_eq!(mul_result.imag.read()?, vec![6.0, 8.0]);

            // Test div_scalar
            let div_result = a.div_scalar(2.0)?;
            assert_eq!(div_result.real.read()?, vec![0.5, 1.0]);
            assert_eq!(div_result.imag.read()?, vec![1.5, 2.0]);

            Ok(())
        }

        #[test]
        fn transcendental_functions() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[PI / 2.0], &device)?;

            // Test sin
            let sin_result = a.sin()?;
            let sin_real = sin_result.real.read()?;
            let sin_imag = sin_result.imag.read()?;
            assert_relative_eq!(sin_real[0], 0.0);

            // Test cos
            let cos_result = a.cos()?;
            let cos_real = cos_result.real.read()?;
            let cos_imag = cos_result.imag.read()?;
            assert_relative_eq!(cos_real[0], 2.5091784786580567);

            // Test sinh
            let sinh_result = a.sinh()?;
            let sinh_real = sinh_result.real.read()?;

            // Test cosh
            let cosh_result = a.cosh()?;
            let cosh_real = cosh_result.real.read()?;
            assert_relative_eq!(cosh_real[0], 0.0);
            Ok(())
        }
    }
}
mod complex_column_vector {
    use crate::ops::*;
    use crate::{ColumnVector, ComplexMatrix, ComplexRowVector, ComplexScalar, RowVector, Scalar};
    use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
    use std::{f64::consts::PI, marker::PhantomData};
    #[derive(Debug, Clone)]
    pub struct ComplexColumnVector<T: WithDType, const ROWS: usize> {
        pub(crate) real: ColumnVector<T, ROWS>,
        pub(crate) imag: ColumnVector<T, ROWS>,
    }
    impl<T: WithDType, const ROWS: usize> IsColumnVector<ROWS> for ComplexColumnVector<T, ROWS> {}

    impl<T: WithDType, const ROWS: usize> TensorBase<T> for ComplexColumnVector<T, ROWS> {
        type ReadOutput = (Vec<T>, Vec<T>);
        #[inline]
        fn device(&self) -> &Device {
            self.real.device()
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
            Ok((self.real.read()?, self.imag.read()?))
        }
    }
    impl<T: WithDType, const ROWS: usize> ElementWiseOp<T> for ComplexColumnVector<T, ROWS> {
        type Output = ComplexColumnVector<T, ROWS>;
        #[inline]
        fn add(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.add(&rhs.real)?,
                imag: self.imag.add(&rhs.imag)?,
            })
        }
        #[inline]
        fn sub(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sub(&rhs.real)?,
                imag: self.imag.sub(&rhs.imag)?,
            })
        }
        #[inline]
        fn mul(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.mul(&rhs.real)?.sub(&self.imag.mul(&rhs.imag)?)?,
                imag: self.real.mul(&rhs.imag)?.add(&self.imag.mul(&rhs.real)?)?,
            })
        }
        #[inline]
        fn div(&self, rhs: &Self) -> Result<Self::Output> {
            let denom = rhs.real.mul(&rhs.real)?.add(&rhs.imag.mul(&rhs.imag)?)?;
            Ok(Self {
                real: self
                    .real
                    .mul(&rhs.real)?
                    .add(&self.imag.mul(&rhs.imag)?)?
                    .div(&denom)?,
                imag: self
                    .imag
                    .mul(&rhs.real)?
                    .sub(&self.real.mul(&rhs.imag)?)?
                    .div(&denom)?,
            })
        }
        #[inline]
        fn clamp(&self, min: &T, max: &T) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clamp(min, max)?,
                imag: self.imag.clamp(min, max)?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize> ScalarOp<T> for ComplexColumnVector<T, ROWS> {
        #[inline]
        fn add_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.add_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn sub_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.sub_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn mul_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.mul_scalar(scalar)?,
                imag: self.imag.mul_scalar(scalar)?,
            })
        }
        #[inline]
        fn div_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.div_scalar(scalar)?,
                imag: self.imag.div_scalar(scalar)?,
            })
        }
        #[inline]
        fn powf(&self, exponent: f64) -> Result<Self> {
            let r = self.magnitude()?;
            let theta = self.arg()?;
            let r_pow = r.powf(exponent)?;
            let new_theta = theta.mul_scalar(T::from_f64(exponent))?;
            Ok(Self {
                real: r_pow.mul(&new_theta.cos()?)?,
                imag: r_pow.mul(&new_theta.sin()?)?,
            })
        }
        #[inline]
        fn pow(&self, other: &Self) -> Result<Self> {
            self.log()?.mul(other)?.exp()
        }
        #[inline]
        fn pow_scalar(&self, scalar: T) -> Result<Self> {
            let r = self.magnitude()?;
            let theta = self.arg()?;
            let r_pow = r.pow_scalar(scalar)?;
            let new_theta = theta.mul_scalar(scalar)?;
            Ok(Self {
                real: r_pow.mul(&new_theta.cos()?)?,
                imag: r_pow.mul(&new_theta.sin()?)?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize> TrigOp<T> for ComplexColumnVector<T, ROWS> {
        type Output = ComplexColumnVector<T, ROWS>;
        #[inline]
        fn sin(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sin()?.mul(&self.imag.cosh()?)?,
                imag: self.real.cos()?.mul(&self.imag.sinh()?)?,
            })
        }
        #[inline]
        fn cos(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cos()?.mul(&self.imag.cosh()?)?,
                imag: self.real.sin()?.mul(&self.imag.sinh()?)?.neg()?,
            })
        }
        #[inline]
        fn sinh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sinh()?.mul(&self.imag.cos()?)?,
                imag: self.real.cosh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn cosh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cosh()?.mul(&self.imag.cos()?)?,
                imag: self.real.sinh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn tanh(&self) -> Result<Self::Output> {
            let two_x = self.real.mul_scalar(T::from_f64(2.0))?;
            let two_y = self.imag.mul_scalar(T::from_f64(2.0))?;

            let sinh_2x = two_x.sinh()?;
            let sin_2y = two_y.sin()?;
            let cosh_2x = two_x.cosh()?;
            let cos_2y = two_y.cos()?;

            let denom = cosh_2x.add(&cos_2y)?;

            Ok(Self {
                real: sinh_2x.div(&denom)?,
                imag: sin_2y.div(&denom)?,
            })
        }
        #[inline]
        fn atan(&self) -> Result<Self::Output> {
            let zero = ColumnVector::zeros(self.device())?;
            let one = ColumnVector::ones(self.device())?;
            let eps = ColumnVector::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi_half = ColumnVector::ones(self.device())?.mul_scalar(T::from_f64(PI / 2.0))?;

            // Form iz
            let iz = ComplexColumnVector {
                real: self.imag.neg()?,
                imag: self.real.clone(),
            };

            // Compute (1 + iz)/(1 - iz)
            let numerator = ComplexColumnVector {
                real: one.clone(),
                imag: zero.clone(),
            }
            .add(&iz)?;

            let denominator = ComplexColumnVector {
                real: one.clone(),
                imag: zero.clone(),
            }
            .sub(&iz)?;

            // Check for special cases
            let mag_real = self.real.abs()?;
            let mag_imag = self.imag.abs()?;

            // z = 0 case
            let is_zero = mag_real.lt(&eps)?.mul(&mag_imag.lt(&eps)?)?;

            // z = ±i case (near branch points)
            let near_i = mag_real
                .lt(&eps)?
                .mul(&(mag_imag.sub(&one)?.abs()?.lt(&eps)?))?;

            // Standard computation
            let standard_result = {
                let ratio = numerator.div(&denominator)?;
                let log_result = ratio.log()?;
                ComplexColumnVector {
                    real: log_result.imag.mul_scalar(T::from_f64(0.5))?,
                    imag: log_result.real.mul_scalar(T::from_f64(-0.5))?,
                }
            };

            // Special case results
            let zero_result = ComplexColumnVector {
                real: zero.clone(),
                imag: zero.clone(),
            };

            let i_result = ComplexColumnVector {
                real: pi_half.mul(&self.imag.div(&mag_imag)?)?,
                imag: ColumnVector::ones(self.device())?.mul_scalar(T::from_f64(f64::INFINITY))?,
            };

            // Combine results using conditional operations
            let result = is_zero.where_cond_complex(
                &zero_result,
                &near_i.where_cond_complex(&i_result, &standard_result)?,
            )?;

            Ok(result)
        }
        #[inline]
        fn atan2(&self, x: &Self) -> Result<Self::Output> {
            let zero = ColumnVector::zeros(self.device())?;
            let eps = ColumnVector::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi = ColumnVector::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

            // Compute magnitudes
            let y_mag = self.magnitude()?;
            let x_mag = x.magnitude()?;

            // Special cases
            let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
            let x_zero = x_mag.lt(&eps)?;

            // Base atan computation
            let z = self.div(x)?;
            let base_atan = z.atan()?;

            // Quadrant adjustments
            let x_neg = x.real.lt(&zero)?;
            let y_gte_zero = self.real.gte(&zero)?;

            // When x < 0: add π for y ≥ 0, subtract π for y < 0
            let adjustment = x_neg.where_cond(&y_gte_zero.where_cond(&pi, &pi.neg()?)?, &zero)?;

            // Apply adjustment to real part only
            let adjusted_result = ComplexColumnVector {
                real: base_atan.real.add(&adjustment)?,
                imag: base_atan.imag,
            };

            // Handle x = 0 cases
            let x_zero_result = ComplexColumnVector {
                real: y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?,
                imag: zero.clone(),
            };

            // Combine all cases
            let result = both_zero.where_cond_complex(
                &ComplexColumnVector {
                    real: zero.clone(),
                    imag: zero.clone(),
                },
                &x_zero.where_cond_complex(&x_zero_result, &adjusted_result)?,
            )?;

            Ok(result)
        }
    }
    impl<T: WithDType, const ROWS: usize> UnaryOp<T> for ComplexColumnVector<T, ROWS> {
        type TransposeOutput = ComplexRowVector<T, ROWS>;
        type ScalarOutput = ComplexScalar<T>;
        #[inline]
        fn neg(&self) -> Result<Self> {
            Ok(Self {
                real: self.real.neg()?,
                imag: self.imag.neg()?,
            })
        }
        #[inline]
        fn abs(&self) -> Result<Self> {
            Ok(Self {
                real: self.magnitude()?,
                imag: ColumnVector::zeros(self.real.device())?,
            })
        }
        #[inline]
        fn exp(&self) -> Result<Self> {
            let exp_real = self.real.exp()?;
            Ok(Self {
                real: exp_real.mul(&self.imag.cos()?)?,
                imag: exp_real.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn log(&self) -> Result<Self> {
            Ok(Self {
                real: self.magnitude()?.log()?,
                imag: self.arg()?,
            })
        }
        #[inline]
        fn mean(&self) -> Result<Self::ScalarOutput> {
            Ok(ComplexScalar {
                real: self.real.mean()?,
                imag: self.imag.mean()?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize> TensorFactory<T> for ComplexColumnVector<T, ROWS> {
        #[inline]
        fn zeros(device: &Device) -> Result<Self> {
            Ok(Self {
                real: ColumnVector::zeros(device)?,
                imag: ColumnVector::zeros(device)?,
            })
        }
        #[inline]
        fn ones(device: &Device) -> Result<Self> {
            Ok(Self {
                real: ColumnVector::ones(device)?,
                imag: ColumnVector::zeros(device)?,
            })
        }
        #[inline]
        fn ones_neg(device: &Device) -> Result<Self> {
            Ok(Self {
                real: ColumnVector::ones_neg(device)?,
                imag: ColumnVector::zeros(device)?,
            })
        }
    }
    impl<F: FloatDType, const ROWS: usize> TensorFactoryFloat<F> for ComplexColumnVector<F, ROWS> {
        #[inline]
        fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: ColumnVector::<F, ROWS>::randn(mean, std, device)?,
                imag: ColumnVector::<F, ROWS>::randn(mean, std, device)?,
            })
        }
        #[inline]
        fn randu(low: F, high: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: ColumnVector::<F, ROWS>::randu(low, high, device)?,
                imag: ColumnVector::<F, ROWS>::randu(low, high, device)?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize> ComplexOp<T> for ComplexColumnVector<T, ROWS> {
        type Output = ComplexColumnVector<T, ROWS>;
        type RealOutput = ColumnVector<T, ROWS>;
        #[inline]
        fn conj(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clone(),
                imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
            })
        }
        #[inline]
        fn magnitude(&self) -> Result<Self::RealOutput> {
            Ok(ColumnVector(
                self.real.0.sqr()?.add(&self.imag.0.sqr()?)?.sqrt()?,
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
    impl<T: WithDType, const ROWS: usize> ColumnVectorOps<T, ROWS> for ComplexColumnVector<T, ROWS> {
        type OuterInput<const COLS: usize> = ComplexRowVector<T, COLS>;
        type OuterOutput<const COLS: usize> = ComplexMatrix<T, ROWS, COLS>;
        type TransposeOutput = ComplexRowVector<T, ROWS>;
        type BroadcastOutput<const COLS: usize> = ComplexMatrix<T, ROWS, COLS>;
        #[inline]
        fn outer<const COLS: usize>(
            &self,
            other: &Self::OuterInput<COLS>,
        ) -> Result<Self::OuterOutput<COLS>> {
            Ok(ComplexMatrix {
                real: self.real.outer(&other.real)?,
                imag: self.imag.outer(&other.imag)?,
            })
        }
        #[inline]
        fn broadcast<const COLS: usize>(&self) -> Result<Self::BroadcastOutput<COLS>> {
            Ok(ComplexMatrix {
                real: self.real.broadcast::<COLS>()?,
                imag: self.imag.broadcast::<COLS>()?,
            })
        }
        #[inline]
        fn transpose(&self) -> Result<Self::TransposeOutput> {
            Ok(ComplexRowVector {
                real: self.real.transpose()?,
                imag: self.imag.transpose()?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize> RealComplexOp<T> for ComplexColumnVector<T, ROWS> {
        type Output = ComplexColumnVector<T, ROWS>;
        type Input = ColumnVector<T, ROWS>;
        #[inline]
        fn mul_complex(&self, rhs: &Self::Input) -> Result<Self::Output> {
            Ok(ComplexColumnVector {
                real: self.real.mul(rhs)?,
                imag: self.imag.mul(rhs)?,
            })
        }
    }
}
mod complex_matrix {
    use crate::ops::*;
    use crate::{ComplexColumnVector, ComplexRowVector, ComplexScalar, Matrix};
    use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
    use std::{f64::consts::PI, marker::PhantomData};
    #[derive(Debug, Clone)]
    pub struct ComplexMatrix<T: WithDType, const ROWS: usize, const C: usize> {
        pub(crate) real: Matrix<T, ROWS, C>,
        pub(crate) imag: Matrix<T, ROWS, C>,
    }
    impl<T: WithDType, const ROWS: usize, const C: usize> IsMatrix<ROWS, C>
        for ComplexMatrix<T, ROWS, C>
    {
    }

    impl<T: WithDType, const ROWS: usize, const COLS: usize> TensorBase<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type ReadOutput = (Vec<Vec<T>>, Vec<Vec<T>>);
        #[inline]
        fn device(&self) -> &Device {
            self.real.device()
        }
        #[inline]
        fn dtype() -> DType {
            T::DTYPE
        }
        #[inline]
        fn shape() -> (usize, usize) {
            (ROWS, COLS)
        }
        #[inline]
        fn read(&self) -> Result<Self::ReadOutput> {
            Ok((self.real.read()?, self.imag.read()?))
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ComplexMatrix<T, ROWS, COLS> {
        pub fn new(
            real: &[T],
            imag: &[T],
            device: &Device,
        ) -> Result<ComplexMatrix<T, ROWS, COLS>> {
            assert!(real.len() == ROWS * COLS);
            assert!(imag.len() == ROWS * COLS);
            Ok(Self {
                real: Matrix::<T, ROWS, COLS>::new(real, device)?,
                imag: Matrix::<T, ROWS, COLS>::new(imag, device)?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ElementWiseOp<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type Output = ComplexMatrix<T, ROWS, COLS>;
        #[inline]
        fn add(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.add(&rhs.real)?,
                imag: self.imag.add(&rhs.imag)?,
            })
        }
        #[inline]
        fn sub(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sub(&rhs.real)?,
                imag: self.imag.sub(&rhs.imag)?,
            })
        }
        #[inline]
        fn mul(&self, rhs: &Self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.mul(&rhs.real)?.sub(&self.imag.mul(&rhs.imag)?)?,
                imag: self.real.mul(&rhs.imag)?.add(&self.imag.mul(&rhs.real)?)?,
            })
        }
        #[inline]
        fn div(&self, rhs: &Self) -> Result<Self::Output> {
            let denom = rhs.real.mul(&rhs.real)?.add(&rhs.imag.mul(&rhs.imag)?)?;
            Ok(Self {
                real: self
                    .real
                    .mul(&rhs.real)?
                    .add(&self.imag.mul(&rhs.imag)?)?
                    .div(&denom)?,
                imag: self
                    .imag
                    .mul(&rhs.real)?
                    .sub(&self.real.mul(&rhs.imag)?)?
                    .div(&denom)?,
            })
        }
        #[inline]
        fn clamp(&self, min: &T, max: &T) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clamp(min, max)?,
                imag: self.imag.clamp(min, max)?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ScalarOp<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        #[inline]
        fn add_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.add_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn sub_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.sub_scalar(scalar)?,
                imag: self.imag.clone(),
            })
        }
        #[inline]
        fn mul_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.mul_scalar(scalar)?,
                imag: self.imag.mul_scalar(scalar)?,
            })
        }
        #[inline]
        fn div_scalar(&self, scalar: T) -> Result<Self> {
            Ok(Self {
                real: self.real.div_scalar(scalar)?,
                imag: self.imag.div_scalar(scalar)?,
            })
        }
        #[inline]
        fn powf(&self, exponent: f64) -> Result<Self> {
            if exponent == 0.0 {
                return Self::ones(self.real.0.device());
            }
            if exponent == 1.0 {
                return Ok(self.clone());
            }
            if exponent == -1.0 {
                let denom = self
                    .real
                    .mul(&self.real)?
                    .add(&self.imag.mul(&self.imag)?)?;
                return Ok(Self {
                    real: self.real.div(&denom)?,
                    imag: self.imag.neg()?.div(&denom)?,
                });
            }
            if exponent.fract() == 0.0 && exponent.abs() <= 4.0 {
                let mut result = self.clone();
                let n = exponent.abs() as i32;
                for _ in 1..n {
                    result = result.mul(self)?;
                }
                if exponent < 0.0 {
                    let denom = result
                        .real
                        .mul(&result.real)?
                        .add(&result.imag.mul(&result.imag)?)?;
                    result = Self {
                        real: result.real.div(&denom)?,
                        imag: result.imag.neg()?.div(&denom)?,
                    };
                }
                return Ok(result);
            }
            let r = self.magnitude()?;
            let ln_r = r.log()?;
            let scaled_ln_r = ln_r.mul_scalar(T::from_f64(exponent))?;
            let r_n = scaled_ln_r.exp()?;
            let theta = self.imag.atan2(&self.real)?;
            let new_theta = theta.mul_scalar(T::from_f64(exponent))?;

            Ok(Self {
                real: r_n.mul(&new_theta.cos()?)?,
                imag: r_n.mul(&new_theta.sin()?)?,
            })
        }
        #[inline]
        fn pow(&self, other: &Self) -> Result<Self> {
            self.log()?.mul(other)?.exp()
        }
        #[inline]
        fn pow_scalar(&self, scalar: T) -> Result<Self> {
            // Handle special cases for scalar integer powers
            if scalar == T::zero() {
                return Self::ones(self.real.device());
            }
            if scalar == T::one() {
                return Ok(self.clone());
            }
            if scalar == T::from_f64(-1.0) {
                let denom = self
                    .real
                    .mul(&self.real)?
                    .add(&self.imag.mul(&self.imag)?)?;
                return Ok(Self {
                    real: self.real.div(&denom)?,
                    imag: self.imag.neg()?.div(&denom)?,
                });
            }

            if scalar.to_f64().fract() == 0.0 && scalar.to_f64().abs() <= 4.0 {
                let mut result = self.clone();
                let n = scalar.to_f64().abs() as i32;
                for _ in 1..n {
                    result = result.mul(self)?;
                }
                if scalar.to_f64() < 0.0 {
                    let denom = result
                        .real
                        .mul(&result.real)?
                        .add(&result.imag.mul(&result.imag)?)?;
                    result = Self {
                        real: result.real.div(&denom)?,
                        imag: result.imag.neg()?.div(&denom)?,
                    };
                }
                return Ok(result);
            }

            let r = self.magnitude()?;
            let ln_r = r.log()?;
            let scaled_ln_r = ln_r.mul_scalar(scalar)?;
            let r_n = scaled_ln_r.exp()?;

            let theta = self.imag.atan2(&self.real)?;
            let new_theta = theta.mul_scalar(scalar)?;

            Ok(Self {
                real: r_n.mul(&new_theta.cos()?)?,
                imag: r_n.mul(&new_theta.sin()?)?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> TrigOp<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type Output = ComplexMatrix<T, ROWS, COLS>;
        #[inline]
        fn sin(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sin()?.mul(&self.imag.cosh()?)?,
                imag: self.real.cos()?.mul(&self.imag.sinh()?)?,
            })
        }
        #[inline]
        fn cos(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cos()?.mul(&self.imag.cosh()?)?,
                imag: self.real.sin()?.mul(&self.imag.sinh()?)?.neg()?,
            })
        }
        #[inline]
        fn sinh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.sinh()?.mul(&self.imag.cos()?)?,
                imag: self.real.cosh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn cosh(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.cosh()?.mul(&self.imag.cos()?)?,
                imag: self.real.sinh()?.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn tanh(&self) -> Result<Self::Output> {
            let two_x = self.real.mul_scalar(T::from_f64(2.0))?;
            let two_y = self.imag.mul_scalar(T::from_f64(2.0))?;

            let sinh_2x = two_x.sinh()?;
            let sin_2y = two_y.sin()?;
            let cosh_2x = two_x.cosh()?;
            let cos_2y = two_y.cos()?;

            let denom = cosh_2x.add(&cos_2y)?;

            Ok(Self {
                real: sinh_2x.div(&denom)?,
                imag: sin_2y.div(&denom)?,
            })
        }
        #[inline]
        fn atan(&self) -> Result<Self::Output> {
            let zero = Matrix::zeros(self.device())?;
            let one = Matrix::ones(self.device())?;
            let eps = Matrix::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi_half = Matrix::ones(self.device())?.mul_scalar(T::from_f64(PI / 2.0))?;

            // Form iz
            let iz = ComplexMatrix {
                real: self.imag.neg()?,
                imag: self.real.clone(),
            };

            // Compute (1 + iz)/(1 - iz)
            let numerator = ComplexMatrix {
                real: one.clone(),
                imag: zero.clone(),
            }
            .add(&iz)?;

            let denominator = ComplexMatrix {
                real: one.clone(),
                imag: zero.clone(),
            }
            .sub(&iz)?;

            // Check for special cases
            let mag_real = self.real.abs()?;
            let mag_imag = self.imag.abs()?;

            // z = 0 case
            let is_zero = mag_real.lt(&eps)?.mul(&mag_imag.lt(&eps)?)?;

            // z = ±i case (near branch points)
            let near_i = mag_real
                .lt(&eps)?
                .mul(&(mag_imag.sub(&one)?.abs()?.lt(&eps)?))?;

            // Standard computation
            let standard_result = {
                let ratio = numerator.div(&denominator)?;
                let log_result = ratio.log()?;
                ComplexMatrix {
                    real: log_result.imag.mul_scalar(T::from_f64(0.5))?,
                    imag: log_result.real.mul_scalar(T::from_f64(-0.5))?,
                }
            };

            // Special case results
            let zero_result = ComplexMatrix {
                real: zero.clone(),
                imag: zero.clone(),
            };

            let i_result = ComplexMatrix {
                real: pi_half.mul(&self.imag.div(&mag_imag)?)?,
                imag: Matrix::ones(self.device())?.mul_scalar(T::from_f64(f64::INFINITY))?,
            };

            // Combine results using conditional operations
            let result = is_zero.where_cond_complex(
                &zero_result,
                &near_i.where_cond_complex(&i_result, &standard_result)?,
            )?;

            Ok(result)
        }
        #[inline]
        fn atan2(&self, x: &Self) -> Result<Self::Output> {
            let zero = Matrix::zeros(self.device())?;
            let eps = Matrix::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
            let pi = Matrix::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

            // Compute magnitudes
            let y_mag = self.magnitude()?;
            let x_mag = x.magnitude()?;

            // Special cases
            let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
            let x_zero = x_mag.lt(&eps)?;

            // Base atan computation
            let z = self.div(x)?;
            let base_atan = z.atan()?;

            // Quadrant adjustments
            let x_neg = x.real.lt(&zero)?;
            let y_gte_zero = self.real.gte(&zero)?;

            // When x < 0: add π for y ≥ 0, subtract π for y < 0
            let adjustment = x_neg.where_cond(&y_gte_zero.where_cond(&pi, &pi.neg()?)?, &zero)?;

            // Apply adjustment to real part only
            let adjusted_result = ComplexMatrix {
                real: base_atan.real.add(&adjustment)?,
                imag: base_atan.imag,
            };

            // Handle x = 0 cases
            let x_zero_result = ComplexMatrix {
                real: y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?,
                imag: zero.clone(),
            };

            // Combine all cases
            let result = both_zero.where_cond_complex(
                &ComplexMatrix {
                    real: zero.clone(),
                    imag: zero.clone(),
                },
                &x_zero.where_cond_complex(&x_zero_result, &adjusted_result)?,
            )?;

            Ok(result)
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> UnaryOp<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type TransposeOutput = ComplexMatrix<T, COLS, ROWS>;
        type ScalarOutput = ComplexScalar<T>;
        #[inline]
        fn neg(&self) -> Result<Self> {
            Ok(Self {
                real: self.real.mul_scalar(T::from_f64(-1.0))?,
                imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
            })
        }
        #[inline]
        fn abs(&self) -> Result<Self> {
            Ok(Self {
                real: self.magnitude()?,
                imag: Matrix::zeros(self.real.device())?,
            })
        }
        #[inline]
        fn exp(&self) -> Result<Self> {
            let exp_real = self.real.exp()?;
            Ok(Self {
                real: exp_real.mul(&self.imag.cos()?)?,
                imag: exp_real.mul(&self.imag.sin()?)?,
            })
        }
        #[inline]
        fn log(&self) -> Result<Self> {
            let magnitude = self.magnitude()?;
            let argument = self.arg()?;

            Ok(Self {
                real: magnitude.log()?,
                imag: argument,
            })
        }
        #[inline]
        fn mean(&self) -> Result<ComplexScalar<T>> {
            Ok(ComplexScalar {
                real: self.real.mean()?,
                imag: self.imag.mean()?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ComparisonOp<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type Output = Matrix<u8, ROWS, COLS>;
        #[inline]
        fn lt(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.lt(&mag_other)
        }
        #[inline]
        fn lte(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.lte(&mag_other)
        }
        #[inline]
        fn eq(&self, other: &Self) -> Result<Self::Output> {
            let real_eq = self.real.eq(&other.real)?;
            let imag_eq = self.imag.eq(&other.imag)?;
            real_eq.mul(&imag_eq)
        }
        #[inline]
        fn ne(&self, other: &Self) -> Result<Self::Output> {
            let real_ne = self.real.ne(&other.real)?;
            let imag_ne = self.imag.ne(&other.imag)?;
            real_ne
                .add(&imag_ne)?
                .gt(&Matrix::<u8, ROWS, COLS>::zeros(self.device())?)
        }
        #[inline]
        fn gt(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.gt(&mag_other)
        }
        #[inline]
        fn gte(&self, other: &Self) -> Result<Self::Output> {
            let mag_self = self.magnitude()?;
            let mag_other = other.magnitude()?;
            mag_self.gte(&mag_other)
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> MatrixOps<T, ROWS, COLS>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type MatMulMatrix<const O: usize> = ComplexMatrix<T, COLS, O>;
        type MatMulOutput<const O: usize> = ComplexMatrix<T, ROWS, O>;
        type RowSumOutput = ComplexColumnVector<T, ROWS>;
        type ColSumOutput = ComplexRowVector<T, COLS>;
        type TransposeOutput = ComplexMatrix<T, COLS, ROWS>;
        #[inline]
        fn sum_rows(&self) -> Result<Self::RowSumOutput> {
            Ok(ComplexColumnVector {
                real: self.real.sum_rows()?,
                imag: self.imag.sum_rows()?,
            })
        }
        #[inline]
        fn sum_cols(&self) -> Result<Self::ColSumOutput> {
            Ok(ComplexRowVector {
                real: self.real.sum_cols()?,
                imag: self.imag.sum_cols()?,
            })
        }
        #[inline]
        fn matmul<const M: usize>(
            &self,
            other: &Self::MatMulMatrix<M>,
        ) -> Result<Self::MatMulOutput<M>> {
            Ok(ComplexMatrix {
                real: self
                    .real
                    .matmul(&other.real)?
                    .sub(&self.imag.matmul(&other.imag)?)?,
                imag: self
                    .real
                    .matmul(&other.imag)?
                    .add(&self.imag.matmul(&other.real)?)?,
            })
        }
        #[inline]
        fn transpose(&self) -> Result<Self::TransposeOutput> {
            Ok(ComplexMatrix {
                real: self.real.transpose()?,
                imag: self.imag.transpose()?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> ComplexOp<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type Output = ComplexMatrix<T, ROWS, COLS>;
        type RealOutput = Matrix<T, ROWS, COLS>;
        #[inline]
        fn conj(&self) -> Result<Self::Output> {
            Ok(Self {
                real: self.real.clone(),
                imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
            })
        }
        #[inline]
        fn magnitude(&self) -> Result<Self::RealOutput> {
            Ok(Matrix(
                self.real.0.sqr()?.add(&self.imag.0.sqr()?)?.sqrt()?,
                PhantomData,
            ))
        }
        #[inline]
        fn arg(&self) -> Result<Self::RealOutput> {
            // Improved arg() implementation with edge case handling:
            let zero = Matrix::zeros(self.device())?;
            let one = Matrix::ones(self.device())?;
            let neg_one = Matrix::ones_neg(self.device())?;
            let pi = Matrix::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
            let pi_half = pi.mul_scalar(T::from_f64(0.5))?;
            let eps = Matrix::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;

            let mag_real = self.real.abs()?;
            let mag_imag = self.imag.abs()?;

            let is_zero = mag_real.lt(&eps)?.mul(&mag_imag.lt(&eps)?)?;
            let near_i = mag_real
                .lt(&eps)?
                .mul(&(mag_imag.sub_scalar(T::one())?.abs()?.lt(&eps)?))?;

            let atan2_result = self.imag.atan2(&self.real)?;

            // Inline sign calculation and edge case handling:
            let arg = is_zero.where_cond(
                &zero,
                &near_i.where_cond(
                    &(self
                        .imag
                        .gt(&zero)?
                        .where_cond(&one, &self.imag.lt(&zero)?.where_cond(&neg_one, &zero)?)?
                        .mul(&pi_half)?),
                    &atan2_result,
                )?,
            )?;

            Ok(arg)
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
    impl<T: WithDType, const ROWS: usize, const COLS: usize> TensorFactory<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        #[inline]
        fn zeros(device: &Device) -> Result<Self> {
            Ok(Self {
                real: Matrix::zeros(device)?,
                imag: Matrix::zeros(device)?,
            })
        }
        #[inline]
        fn ones(device: &Device) -> Result<Self> {
            Ok(Self {
                real: Matrix::ones(device)?,
                imag: Matrix::zeros(device)?,
            })
        }
        #[inline]
        fn ones_neg(device: &Device) -> Result<Self> {
            Ok(Self {
                real: Matrix::ones_neg(device)?,
                imag: Matrix::zeros(device)?,
            })
        }
    }
    impl<F: FloatDType, const ROWS: usize, const COLS: usize> TensorFactoryFloat<F>
        for ComplexMatrix<F, ROWS, COLS>
    {
        #[inline]
        fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: Matrix::<F, ROWS, COLS>::randn(mean, std, device)?,
                imag: Matrix::<F, ROWS, COLS>::randn(mean, std, device)?,
            })
        }
        #[inline]
        fn randu(low: F, high: F, device: &Device) -> Result<Self> {
            Ok(Self {
                real: Matrix::<F, ROWS, COLS>::randu(low, high, device)?,
                imag: Matrix::<F, ROWS, COLS>::randu(low, high, device)?,
            })
        }
    }
    impl<T: WithDType, const ROWS: usize, const COLS: usize> MatrixFactory<T, ROWS, COLS>
        for ComplexMatrix<T, ROWS, COLS>
    {
        #[inline]
        fn eye(device: &Device) -> Result<Self> {
            Ok(ComplexMatrix {
                real: Matrix::eye(device)?,
                imag: Matrix::zeros(device)?,
            })
        }
    }

    impl<T: WithDType, const ROWS: usize, const COLS: usize> RealComplexOp<T>
        for ComplexMatrix<T, ROWS, COLS>
    {
        type Output = ComplexMatrix<T, ROWS, COLS>;
        type Input = Matrix<T, ROWS, COLS>;
        #[inline]
        fn mul_complex(&self, rhs: &Self::Input) -> Result<Self::Output> {
            Ok(ComplexMatrix {
                real: self.real.mul(rhs)?,
                imag: self.imag.mul(rhs)?,
            })
        }
    }
    #[cfg(test)]
    mod test {
        use crate::*;
        use approx::assert_relative_eq;
        use candle_core::{Device, Result};
        use std::f64::consts::PI;

        #[test]
        fn new_complex_tensor() -> Result<()> {
            let device = Device::Cpu;
            let real = vec![1.0, 2.0, 3.0, 4.0];
            let imag = vec![5.0, 6.0, 7.0, 8.0];
            let tensor = ComplexMatrix::<f64, 2, 2>::new(&real, &imag, &device)?;
            assert_eq!(tensor.real.read()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            assert_eq!(tensor.imag.read()?, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
            Ok(())
        }

        #[test]
        fn sub() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 2.0, 3.0, 4.0],
                &[5.0, 6.0, 7.0, 8.0],
                &device,
            )?;
            let b = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 1.0, 1.0, 1.0],
                &[1.0, 1.0, 1.0, 1.0],
                &device,
            )?;
            let c = a.sub(&b)?;
            assert_eq!(c.real.read()?, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
            assert_eq!(c.imag.read()?, vec![vec![4.0, 5.0], vec![6.0, 7.0]]);
            Ok(())
        }

        #[test]
        fn mul() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 2.0, 3.0, 4.0],
                &[1.0, 1.0, 1.0, 1.0],
                &device,
            )?;
            let b = ComplexMatrix::<f64, 2, 2>::new(
                &[2.0, 2.0, 2.0, 2.0],
                &[1.0, 1.0, 1.0, 1.0],
                &device,
            )?;
            let c = a.mul(&b)?;

            // Correct calculations:
            // (1+i)(2+i) = (1*2 - 1*1) + (1*1 + 1*2)i = 1 + 3i
            // (2+i)(2+i) = (2*2 - 1*1) + (2*1 + 1*2)i = 3 + 4i
            // (3+i)(2+i) = (3*2 - 1*1) + (3*1 + 1*2)i = 5 + 5i
            // (4+i)(2+i) = (4*2 - 1*1) + (4*1 + 1*2)i = 7 + 6i

            assert_eq!(c.real.read()?, vec![vec![1.0, 3.0], vec![5.0, 7.0]]);
            assert_eq!(c.imag.read()?, vec![vec![3.0, 4.0], vec![5.0, 6.0]]); // Corrected imaginary parts
            Ok(())
        }
        #[test]
        fn div() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 1, 1>::new(&[4.0], &[0.0], &device)?;
            let b = ComplexMatrix::<f64, 1, 1>::new(&[2.0], &[0.0], &device)?;
            let c = a.div(&b)?;
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0][0], 2.0);
            assert_relative_eq!(imag[0][0], 0.0);
            Ok(())
        }

        #[test]
        fn div_pure_imag() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
            let b = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
            let c = a.div(&b)?;
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0][0], 1.0);
            assert_relative_eq!(imag[0][0], 0.0);
            Ok(())
        }

        #[test]
        fn conj() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 2.0, 3.0, 4.0],
                &[5.0, 6.0, 7.0, 8.0],
                &device,
            )?;
            let c = a.conj()?;
            assert_eq!(c.real.read()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            assert_eq!(c.imag.read()?, vec![vec![-5.0, -6.0], vec![-7.0, -8.0]]);
            Ok(())
        }

        #[test]
        fn abs() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 1>::new(&[3.0, 0.0], &[4.0, 1.0], &device)?;
            let c = a.abs()?;
            // |3 + 4i| = 5, |0 + i| = 1
            let abs = c.real.read()?;
            assert_relative_eq!(abs[0][0], 5.0);
            assert_relative_eq!(abs[1][0], 1.0);
            Ok(())
        }

        #[test]
        fn log() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
            let c = a.log()?;
            // ln(i) = iπ/2
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0][0], 0.0);
            assert_relative_eq!(imag[0][0], PI / 2.0);
            Ok(())
        }

        #[test]
        fn pow_scalar() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 1, 1>::new(&[1.0], &[1.0], &device)?; // 1 + i
            let c = a.pow_scalar(2.0)?;
            // (1+i)^2 = 0 + 2i
            let real = c.real.read()?;
            let imag = c.imag.read()?;

            assert_relative_eq!(real[0][0], 0.0);
            assert_relative_eq!(imag[0][0], 2.0);
            Ok(())
        }
        #[test]
        fn powf_edge_cases() -> Result<()> {
            let device = Device::Cpu;
            // Test case 1: Negative base, fractional exponent
            let tensor = RowVector::<f64, 1>::new(&[-1.0], &device)?;
            let result = tensor.powf(0.5)?; // Should be NaN or complex depending on your backend
                                            // Add assertions based on expected behavior (NaN, complex number, or exception)

            // Test case 2: Zero base, positive exponent
            let tensor = RowVector::<f64, 1>::new(&[0.0], &device)?;
            let result = tensor.powf(2.0)?;
            assert_eq!(result.read()?, vec![0.0]);

            // Test case 3: Zero base, negative exponent
            let tensor = RowVector::<f64, 1>::new(&[0.0], &device)?;
            let result = tensor.powf(-2.0)?; // Should be infinity or exception
                                             // Add assertions based on expected behavior

            Ok(())
        }
        #[test]
        fn pow() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
            let b = ComplexMatrix::<f64, 1, 1>::new(&[2.0], &[0.0], &device)?;
            let c = ComplexOp::pow(&a, &b)?;
            // i^2 = -1
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0][0], -1.0);
            assert_relative_eq!(imag[0][0], 0.0);
            Ok(())
        }

        #[test]
        fn sub_scalar() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 2.0, 3.0, 4.0],
                &[5.0, 6.0, 7.0, 8.0],
                &device,
            )?;
            let result = a.sub_scalar(1.0)?;
            assert_eq!(result.real.read()?, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
            assert_eq!(result.imag.read()?, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
            Ok(())
        }

        #[test]
        fn div_scalar() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[2.0, 4.0, 6.0, 8.0],
                &[10.0, 12.0, 14.0, 16.0],
                &device,
            )?;
            let result = a.div_scalar(2.0)?;
            assert_eq!(result.real.read()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            assert_eq!(result.imag.read()?, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
            Ok(())
        }

        #[test]
        fn zeros() -> Result<()> {
            let device = Device::Cpu;
            let zeros = ComplexMatrix::<f64, 2, 2>::zeros(&device)?;
            assert_eq!(zeros.real.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
            assert_eq!(zeros.imag.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
            Ok(())
        }

        #[test]
        fn ones() -> Result<()> {
            let device = Device::Cpu;
            let ones = ComplexMatrix::<f64, 2, 2>::ones(&device)?;
            assert_eq!(ones.real.read()?, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
            assert_eq!(ones.imag.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
            Ok(())
        }

        #[test]
        fn ones_neg() -> Result<()> {
            let device = Device::Cpu;
            let ones_neg = ComplexMatrix::<f64, 2, 2>::ones_neg(&device)?;
            assert_eq!(
                ones_neg.real.read()?,
                vec![vec![-1.0, -1.0], vec![-1.0, -1.0]]
            );
            assert_eq!(ones_neg.imag.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
            Ok(())
        }

        #[test]
        fn add() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 2.0, 3.0, 4.0],
                &[5.0, 6.0, 7.0, 8.0],
                &device,
            )?;
            let b = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 1.0, 1.0, 1.0],
                &[1.0, 1.0, 1.0, 1.0],
                &device,
            )?;
            let c = a.add(&b)?;
            assert_eq!(c.real.read()?, vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
            assert_eq!(c.imag.read()?, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
            Ok(())
        }

        #[test]
        fn matmul() -> Result<()> {
            let device = Device::Cpu;
            // (1+i)(2+2i) + (2+2i)(3+3i) = (2-2) + (2+2)i + (6-6) + (6+6)i = 0 + 16i
            let a = ComplexMatrix::<f64, 1, 2>::new(&[1.0, 2.0], &[1.0, 2.0], &device)?;
            let b = ComplexMatrix::<f64, 2, 1>::new(&[2.0, 3.0], &[2.0, 3.0], &device)?;
            let c = a.matmul(&b)?;
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0][0], 0.0);
            assert_relative_eq!(imag[0][0], 16.0);
            Ok(())
        }

        #[test]
        fn transpose() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 2.0, 3.0, 4.0],
                &[5.0, 6.0, 7.0, 8.0],
                &device,
            )?;
            let c = a.transpose()?;
            assert_eq!(c.real.read()?, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
            assert_eq!(c.imag.read()?, vec![vec![5.0, 7.0], vec![6.0, 8.0]]);
            Ok(())
        }

        #[test]
        fn exp() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[PI / 2.0], &device)?;
            let c = a.exp()?;
            // e^(iπ/2) = i
            let real = c.real.read()?;
            let imag = c.imag.read()?;
            assert_relative_eq!(real[0][0], 0.0);
            assert_relative_eq!(imag[0][0], 1.0);
            Ok(())
        }

        #[test]
        fn scalar_operations() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 2.0, 3.0, 4.0],
                &[5.0, 6.0, 7.0, 8.0],
                &device,
            )?;

            // Test add_scalar
            let add_result = a.add_scalar(2.0)?;
            assert_eq!(
                add_result.real.read()?,
                vec![vec![3.0, 4.0], vec![5.0, 6.0]]
            );
            assert_eq!(
                add_result.imag.read()?,
                vec![vec![5.0, 6.0], vec![7.0, 8.0]]
            );

            // Test mul_scalar
            let mul_result = a.mul_scalar(2.0)?;
            assert_eq!(
                mul_result.real.read()?,
                vec![vec![2.0, 4.0], vec![6.0, 8.0]]
            );
            assert_eq!(
                mul_result.imag.read()?,
                vec![vec![10.0, 12.0], vec![14.0, 16.0]]
            );

            Ok(())
        }

        #[test]
        fn transcendental_functions() -> Result<()> {
            let device = Device::Cpu;
            let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[PI / 2.0], &device)?;

            // Test sin
            let sin_result = a.sin()?;
            let sin_real = sin_result.real.read()?;
            let sin_imag = sin_result.imag.read()?;
            assert_relative_eq!(sin_real[0][0], 0.0);

            // Test cos
            let cos_result = a.cos()?;
            let cos_real = cos_result.real.read()?;
            let cos_imag = cos_result.imag.read()?;
            assert_relative_eq!(cos_real[0][0], 2.5091784786580567);

            // Test sinh
            let sinh_result = a.sinh()?;
            let sinh_real = sinh_result.real.read()?;
            assert_relative_eq!(sinh_real[0][0], 0.0);

            // Test cosh
            let cosh_result = a.cosh()?;
            let cosh_real = cosh_result.real.read()?;
            assert_relative_eq!(cosh_real[0][0], 0.0);

            Ok(())
        }

        #[test]
        fn where_cond() -> Result<()> {
            let device = Device::Cpu;
            let cond = Matrix::<u8, 2, 2>::new(&[1, 0, 1, 0], &device)?;
            let on_true = ComplexMatrix::<f64, 2, 2>::new(
                &[1.0, 1.0, 1.0, 1.0],
                &[1.0, 1.0, 1.0, 1.0],
                &device,
            )?;
            let on_false = ComplexMatrix::<f64, 2, 2>::new(
                &[2.0, 2.0, 2.0, 2.0],
                &[2.0, 2.0, 2.0, 2.0],
                &device,
            )?;
            let result = cond.where_cond_complex(&on_true, &on_false)?;
            assert_eq!(result.real.read()?, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
            assert_eq!(result.imag.read()?, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
            Ok(())
        }
    }
}
pub use {
    column_vector::ColumnVector, complex_column_vector::ComplexColumnVector,
    complex_matrix::ComplexMatrix, complex_row_vector::ComplexRowVector,
    complex_scalar::ComplexScalar, matrix::Matrix, row_vector::RowVector, scalar::Scalar,
};

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
