use crate::ops::*;
use crate::ComplexScalar;
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct Scalar<T: WithDType>(pub(crate) Tensor, pub(crate) PhantomData<T>);

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
impl_elementwise_op!(Scalar, 0);
impl_scalar_op!(Scalar, 0);
impl_trig_op!(Scalar, 0);
impl_unary_op!(Scalar, 0, Scalar);
impl_comparison_op!(Scalar, 0, Scalar);

impl<T: WithDType> ConditionalOp<T> for Scalar<u8> {
    type Output = Scalar<T>;
    type ComplexOutput = ComplexScalar<T>;

    #[inline]
    fn where_cond(&self, on_true: &Self::Output, on_false: &Self::Output) -> Result<Self::Output> {
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
