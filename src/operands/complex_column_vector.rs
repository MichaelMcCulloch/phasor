use crate::ops::*;
use crate::utils::methods::generic_complex_outer;
use crate::{ColumnVector, ComplexMatrix, ComplexRowVector};
use candle_core::{DType, Device, FloatDType, Result, WithDType};
use std::marker::PhantomData;

use super::Matrix;
#[derive(Debug, Clone)]
pub struct ComplexColumnVector<T: WithDType, const ROWS: usize> {
    pub(crate) real: ColumnVector<T, ROWS>,
    pub(crate) imag: ColumnVector<T, ROWS>,
}

impl_complex_op!(ComplexColumnVector, real, imag, ColumnVector, ROWS);
impl_complex_elementwise_op!(ComplexColumnVector, real, imag, ColumnVector, ROWS);
impl_complex_scalar_op!(ComplexColumnVector, real, imag, ColumnVector, ROWS);
impl_complex_trig_op!(ComplexColumnVector, real, imag, ColumnVector, ROWS);
impl_complex_comparison_op!(ComplexColumnVector, real, imag, ColumnVector, ROWS);
impl_complex_tensor_factory!(ComplexColumnVector, real, imag, ColumnVector, ROWS);
impl_complex_tensor_factory_float!(ComplexColumnVector, real, imag, ColumnVector, ROWS);

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
impl<T: WithDType, const ROWS: usize> ComplexColumnVector<T, ROWS> {
    pub fn new(
        data_real: &[T],
        data_imag: &[T],
        device: &Device,
    ) -> Result<ComplexColumnVector<T, ROWS>> {
        assert!(data_real.len() == ROWS);
        assert!(data_imag.len() == ROWS);
        Ok(Self {
            real: ColumnVector::new(data_real, device)?,
            imag: ColumnVector::new(data_imag, device)?,
        })
    }
}
impl<T: WithDType, const ROWS: usize> ColumnVectorOps<T, ROWS> for ComplexColumnVector<T, ROWS> {
    type OuterInput<const COLS: usize> = ComplexRowVector<T, COLS>;
    type OuterOutput<const COLS: usize> = ComplexMatrix<T, ROWS, COLS>;
    type TransposeOutput = ComplexRowVector<T, ROWS>;
    type BroadcastOutput<const C: usize> = ComplexMatrix<T, ROWS, C>;
    #[inline]
    fn outer<const COLS: usize>(
        &self,
        other: &Self::OuterInput<COLS>,
    ) -> Result<Self::OuterOutput<COLS>> {
        let (real_part, imag_part) =
            generic_complex_outer::<T>(&self.real.0, &self.imag.0, &other.real.0, &other.imag.0)?;

        Ok(ComplexMatrix {
            real: Matrix(real_part, PhantomData),
            imag: Matrix(imag_part, PhantomData),
        })
    }

    #[inline]
    fn transpose(&self) -> Result<Self::TransposeOutput> {
        Ok(ComplexRowVector {
            real: self.real.transpose()?,
            imag: self.imag.transpose()?,
        })
    }
    #[inline]
    fn broadcast<const C: usize>(&self) -> Result<Self::BroadcastOutput<C>> {
        Ok(ComplexMatrix {
            real: self.real.broadcast::<C>()?,
            imag: self.imag.broadcast::<C>()?,
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
