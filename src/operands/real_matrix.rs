use crate::*;
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct Matrix<T: WithDType, const ROWS: usize, const C: usize>(
    pub(crate) Tensor,
    pub(crate) PhantomData<T>,
);
impl_tensor_base!(Matrix, 0, ROWS, COLS);
impl_elementwise_op!(Matrix, 0, ROWS, COLS);
impl_scalar_op!(Matrix, 0, ROWS, COLS);
impl_trig_op!(Matrix, 0, ROWS, COLS);
impl_unary_op!(Matrix, 0, Matrix, COLS, ROWS);
impl_comparison_op!(Matrix, 0, Matrix, ROWS, COLS);
impl_tensor_factory!(Matrix, 0, ROWS, COLS);
impl_tensor_factory_float!(Matrix, 0, ROWS, COLS);
impl_conditional_op!(Matrix, 0, Matrix, ComplexMatrix, ROWS, COLS);
impl_boolean_op!(Matrix, 0, ROWS, COLS);

impl<T: WithDType, const ROWS: usize, const C: usize> IsMatrix<ROWS, C> for Matrix<T, ROWS, C> {}

impl<T: WithDType, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    pub fn new(data: &[T], device: &Device) -> Result<Matrix<T, ROWS, COLS>> {
        assert!(data.len() == ROWS * COLS);
        Ok(Self(
            Tensor::from_slice(data, (ROWS, COLS), device)?,
            PhantomData,
        ))
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
