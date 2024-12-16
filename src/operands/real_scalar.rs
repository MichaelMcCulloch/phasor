use crate::ops::*;
use crate::ComplexScalar;
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct Scalar<T: WithDType>(pub(crate) Tensor, pub(crate) PhantomData<T>);

impl_tensor_base!(Scalar, 0);
impl_elementwise_op!(Scalar, 0);
impl_scalar_op!(Scalar, 0);
impl_power_op!(Scalar, 0);
impl_trig_op!(Scalar, 0);
impl_unary_op!(Scalar, 0);
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
