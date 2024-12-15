use crate::ops::*;
use crate::Scalar;
use candle_core::{DType, Device, FloatDType, Result, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct ComplexScalar<T: WithDType> {
    pub(crate) real: Scalar<T>,
    pub(crate) imag: Scalar<T>,
}

impl_complex_op!(ComplexScalar, real, imag, Scalar);
impl_complex_elementwise_op!(ComplexScalar, real, imag, Scalar);
impl_complex_scalar_op!(ComplexScalar, real, imag, Scalar);
impl_complex_trig_op!(ComplexScalar, real, imag, Scalar);
impl_complex_unary_op!(ComplexScalar, real, imag, Scalar, ComplexScalar);
impl_complex_comparison_op!(ComplexScalar, real, imag, Scalar);
impl_complex_tensor_factory!(ComplexScalar, real, imag, Scalar);
impl_complex_tensor_factory_float!(ComplexScalar, real, imag, Scalar);

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
