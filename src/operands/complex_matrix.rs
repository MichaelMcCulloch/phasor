use crate::ops::*;
use crate::{ComplexColumnVector, ComplexRowVector, Matrix};
use candle_core::{DType, Device, FloatDType, Result, WithDType};
use std::marker::PhantomData;
#[derive(Debug, Clone)]
pub struct ComplexMatrix<T: WithDType, const ROWS: usize, const C: usize> {
    pub(crate) real: Matrix<T, ROWS, C>,
    pub(crate) imag: Matrix<T, ROWS, C>,
}

impl_complex_op!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);
impl_complex_elementwise_op!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);
impl_complex_scalar_op!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);
impl_complex_trig_op!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);
impl_complex_comparison_op!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);
impl_complex_tensor_factory!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);
impl_complex_tensor_factory_float!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);

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
    pub fn new(real: &[T], imag: &[T], device: &Device) -> Result<ComplexMatrix<T, ROWS, COLS>> {
        assert!(real.len() == ROWS * COLS);
        assert!(imag.len() == ROWS * COLS);
        Ok(Self {
            real: Matrix::<T, ROWS, COLS>::new(real, device)?,
            imag: Matrix::<T, ROWS, COLS>::new(imag, device)?,
        })
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
        let (real, imag) = crate::utils::methods::generic_complex_matmul::<T>(
            &self.real.0,
            &self.imag.0,
            &other.real.0,
            &other.imag.0,
        )?;
        Ok(ComplexMatrix {
            real: Matrix(real, PhantomData),
            imag: Matrix(imag, PhantomData),
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
