use crate::ops::*;
use crate::ComplexScalar;
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct Scalar<T: WithDType>(pub(crate) Tensor, pub(crate) PhantomData<T>);

impl_tensor_base!(Scalar, 0);
impl_elementwise_op!(Scalar, 0);
impl_scalar_op!(Scalar, 0);
impl_trig_op!(Scalar, 0);
impl_unary_op!(Scalar, 0, Scalar);
impl_comparison_op!(Scalar, 0, Scalar);
impl_tensor_factory!(Scalar, 0);
impl_tensor_factory_float!(Scalar, 0);
impl_conditional_op!(Scalar, 0, Scalar, ComplexScalar);
impl_boolean_op!(Scalar, 0);

impl<T: WithDType> IsScalar for Scalar<T> {}
impl<T: WithDType> Scalar<T> {
    pub fn new(data: T, device: &Device) -> Result<Scalar<T>> {
        Ok(Self(
            Tensor::from_slice(&[data], Self::shape(), device)?,
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
