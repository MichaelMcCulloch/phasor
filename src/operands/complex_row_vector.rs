// In src/operands/complex_row_vector.rs

use crate::ops::*;
use crate::{ComplexColumnVector, ComplexMatrix, ComplexScalar, RowVector, Scalar};
use candle_core::{DType, Device, FloatDType, Result, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct ComplexRowVector<T: WithDType, const ROWS: usize> {
    pub(crate) real: RowVector<T, ROWS>,
    pub(crate) imag: RowVector<T, ROWS>,
}

impl_complex_op!(ComplexRowVector, real, imag, RowVector, ROWS);
impl_complex_elementwise_op!(ComplexRowVector, real, imag, RowVector, ROWS);
impl_complex_scalar_op!(ComplexRowVector, real, imag, RowVector, ROWS);
impl_complex_trig_op!(ComplexRowVector, real, imag, RowVector, ROWS);
impl_complex_unary_op!(
    ComplexRowVector,
    real,
    imag,
    RowVector,
    ComplexColumnVector,
    ROWS
);
impl_complex_comparison_op!(ComplexRowVector, real, imag, RowVector, ROWS);
impl_complex_tensor_factory!(ComplexRowVector, real, imag, RowVector, ROWS);
impl_complex_tensor_factory_float!(ComplexRowVector, real, imag, RowVector, ROWS);

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
