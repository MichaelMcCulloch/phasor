use crate::*;
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::{f64::consts::PI, marker::PhantomData};

#[derive(Debug, Clone)]
pub struct ColumnVector<T: WithDType, const ROWS: usize>(
    pub(crate) Tensor,
    pub(crate) PhantomData<T>,
);
impl_tensor_base!(ColumnVector, 0, ROWS);
impl_elementwise_op!(ColumnVector, 0, ROWS);
impl_scalar_op!(ColumnVector, 0, ROWS);
impl_trig_op!(ColumnVector, 0, ROWS);
impl_unary_op!(ColumnVector, 0, RowVector, ROWS);
impl_comparison_op!(ColumnVector, 0, ColumnVector, ROWS);
impl_tensor_factory!(ColumnVector, 0, ROWS);
impl_tensor_factory_float!(ColumnVector, 0, ROWS);
impl_conditional_op!(ColumnVector, 0, ColumnVector, ComplexColumnVector, ROWS);
impl_boolean_op!(ColumnVector, 0, ROWS);

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
