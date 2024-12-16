use crate::ops::*;
use crate::*;
use crate::{ColumnVector, ComplexRowVector};
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::marker::PhantomData;
use utils::methods::{generic_broadcast, generic_dot, generic_transpose};
#[derive(Debug, Clone)]
pub struct RowVector<T: WithDType, const ROWS: usize>(pub(crate) Tensor, pub(crate) PhantomData<T>);

impl_tensor_base!(RowVector, 0, ROWS);
impl_elementwise_op!(RowVector, 0, ROWS);
impl_scalar_op!(RowVector, 0, ROWS);
impl_power_op!(RowVector, 0, ROWS);
impl_trig_op!(RowVector, 0, ROWS);
impl_unary_op!(RowVector, 0, ROWS);
impl_comparison_op!(RowVector, 0, RowVector, ROWS);
impl_tensor_factory!(RowVector, 0, ROWS);
impl_tensor_factory_float!(RowVector, 0, ROWS);
impl_conditional_op!(RowVector, 0, RowVector, ComplexRowVector, ROWS);
impl_boolean_op!(RowVector, 0, ROWS);

impl<T: WithDType, const ROWS: usize> IsRowVector<ROWS> for RowVector<T, ROWS> {}
impl<T: WithDType, const COLS: usize> RowVector<T, COLS> {
    pub fn new(data: &[T], device: &Device) -> Result<RowVector<T, COLS>> {
        assert!(data.len() == COLS);
        Ok(Self(
            Tensor::from_slice(data, (1, COLS), device)?,
            PhantomData,
        ))
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
        Ok(Scalar(generic_dot::<T>(&self.0, &other.0)?, PhantomData))
    }
    #[inline]
    fn matmul<const M: usize>(
        &self,
        other: &Self::MatMulMatrix<M>,
    ) -> Result<Self::MatMulOutput<M>> {
        Ok(RowVector(
            crate::utils::methods::generic_matmul::<T>(&self.0, &other.0)?,
            PhantomData,
        ))
    }
    #[inline]
    fn transpose(&self) -> Result<Self::TransposeOutput> {
        Ok(ColumnVector(generic_transpose::<T>(&self.0)?, PhantomData))
    }
    #[inline]
    fn broadcast<const ROWS: usize>(&self) -> Result<Self::BroadcastOutput<ROWS>> {
        Ok(Matrix(
            generic_broadcast::<T>(&self.0, (ROWS, COLS))?,
            PhantomData,
        ))
    }
}
