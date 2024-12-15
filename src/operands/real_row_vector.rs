use crate::ops::*;
use crate::*;
use crate::{ColumnVector, ComplexRowVector};
use candle_core::{DType, Device, FloatDType, Result, Tensor, WithDType};
use std::marker::PhantomData;
#[derive(Debug, Clone)]
pub struct RowVector<T: WithDType, const ROWS: usize>(pub(crate) Tensor, pub(crate) PhantomData<T>);

impl_tensor_base!(RowVector, 0, ROWS);
impl_elementwise_op!(RowVector, 0, ROWS);
impl_scalar_op!(RowVector, 0, ROWS);
impl_trig_op!(RowVector, 0, ROWS);
impl_unary_op!(RowVector, 0, ColumnVector, ROWS);
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
        Ok(Scalar(
            self.0.mul(&other.0)?.sum_all()?.reshape((1, 1))?,
            PhantomData,
        ))
    }
    #[inline]
    fn matmul<const M: usize>(
        &self,
        other: &Self::MatMulMatrix<M>,
    ) -> Result<Self::MatMulOutput<M>> {
        Ok(RowVector(self.0.matmul(&other.0)?, PhantomData))
    }
    #[inline]
    fn transpose(&self) -> Result<Self::TransposeOutput> {
        Ok(ColumnVector(self.0.t()?, PhantomData))
    }
    #[inline]
    fn broadcast<const ROWS: usize>(&self) -> Result<Self::BroadcastOutput<ROWS>> {
        Ok(Matrix(self.0.broadcast_as((ROWS, COLS))?, PhantomData))
    }
}

#[cfg(test)]
mod test {

    use crate::*;
    use approx::assert_relative_eq;
    use candle_core::{DType, Device, Result};
    use operands::test::assert_relative_eq_vec;

    #[test]
    fn new_tensor1d() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let tensor = RowVector::<f64, 3>::new(&data, &device)?;
        assert_eq!(tensor.0.shape().dims(), [1, 3]);
        Ok(())
    }

    #[test]
    fn zeros() -> Result<()> {
        let device = Device::Cpu;
        let zeros = RowVector::<f64, 3>::zeros(&device)?;
        assert_eq!(zeros.read()?, vec![0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn ones() -> Result<()> {
        let device = Device::Cpu;
        let ones = RowVector::<f64, 3>::ones(&device)?;
        assert_eq!(ones.read()?, vec![1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn ones_neg() -> Result<()> {
        let device = Device::Cpu;
        let ones_neg = RowVector::<f64, 3>::ones_neg(&device)?;
        assert_eq!(ones_neg.read()?, vec![-1.0, -1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn add_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let added = tensor.add_scalar(2.0)?;
        assert_eq!(added.read()?, vec![3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn sub_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let subbed = tensor.sub_scalar(1.0)?;
        assert_eq!(subbed.read()?, vec![0.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn mul_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let mulled = tensor.mul_scalar(3.0)?;
        assert_eq!(mulled.read()?, vec![3.0, 6.0, 9.0]);
        Ok(())
    }

    #[test]
    fn div_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let divided = tensor.div_scalar(2.0)?;
        assert_eq!(divided.read()?, vec![0.5, 1.0, 1.5]);
        Ok(())
    }

    #[test]
    fn pow_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let powered = tensor.pow_scalar(2.0)?;
        assert_eq!(powered.read()?, vec![1.0, 4.0, 9.0]);
        Ok(())
    }

    #[test]
    fn element_wise_add() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
        let added = tensor1.add(&tensor2)?;
        assert_eq!(added.read()?, vec![3.0, 5.0, 7.0]);
        Ok(())
    }

    #[test]
    fn element_wise_sub() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
        let subbed = tensor1.sub(&tensor2)?;
        assert_eq!(subbed.read()?, vec![-1.0, -1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn element_wise_mul() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
        let mulled = tensor1.mul(&tensor2)?;
        assert_eq!(mulled.read()?, vec![2.0, 6.0, 12.0]);
        Ok(())
    }

    #[test]
    fn element_wise_div() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
        let divided = tensor1.div(&tensor2)?;
        assert_eq!(divided.read()?, vec![0.5, 2.0 / 3.0, 0.75]);
        Ok(())
    }

    #[test]
    fn exp() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let exp = tensor.exp()?;
        let exp_vec = exp.read()?;
        assert_relative_eq!(exp_vec[0], 2.7182817);
        Ok(())
    }

    #[test]
    fn log() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let log = tensor.log()?;
        let log_vec = log.read()?;
        assert_relative_eq!(log_vec[0], 1.0f64.ln());
        assert_relative_eq!(log_vec[1], 2.0f64.ln());
        assert_relative_eq!(log_vec[2], 3.0f64.ln());
        Ok(())
    }

    #[test]
    fn cos() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let cos = tensor.cos()?;
        let cos_vec = cos.read()?;
        assert_relative_eq!(cos_vec[0], 0.5403023);
        Ok(())
    }
    #[test]
    fn sin() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let sin = tensor.sin()?;
        let sin_vec = sin.read()?;
        assert_relative_eq!(sin_vec[0], 0.8414709);
        Ok(())
    }

    #[test]
    fn dot_product() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
        let dot = tensor1.dot(&tensor2)?;
        assert_relative_eq!(dot.read()?, 20.0);
        Ok(())
    }

    #[test]
    fn outer_product() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = ColumnVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let tensor2 = RowVector::<f64, 3>::new(&[2.0, 3.0, 4.0], &device)?;
        let outer = tensor1.outer(&tensor2)?;
        let outer_vec = outer.read()?;
        assert_eq!(outer_vec[0], vec![2.0, 3.0, 4.0]);
        assert_eq!(outer_vec[1], vec![4.0, 6.0, 8.0]);
        assert_eq!(outer_vec[2], vec![6.0, 9.0, 12.0]);
        Ok(())
    }

    #[test]
    fn where_condition() -> Result<()> {
        let device = Device::Cpu;
        let cond = RowVector::<u8, 3>::new(&[1, 0, 1], &device)?;
        let on_true = RowVector::<u8, 3>::new(&[1, 1, 1], &device)?;
        let on_false = RowVector::<u8, 3>::new(&[2, 2, 2], &device)?;
        let result = cond.where_cond(&on_true, &on_false)?;
        assert_eq!(result.read()?, vec![1, 2, 1]);
        Ok(())
    }
    #[test]
    fn neg() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let neg = tensor.neg()?;
        assert_eq!(neg.read()?, vec![-1.0, -2.0, -3.0]);
        Ok(())
    }

    #[test]
    fn abs() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[-1.0, 2.0, -3.0], &device)?;
        let abs = tensor.abs()?;
        assert_eq!(abs.read()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn tanh() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[0.0, 1.0, 2.0], &device)?;
        let tanh = tensor.tanh()?;
        let tanh_vec = tanh.read()?;
        assert_relative_eq!(tanh_vec[0], 0.0);
        assert_relative_eq!(tanh_vec[1], 0.7615942);
        assert_relative_eq!(tanh_vec[2], 0.9640276);
        Ok(())
    }

    #[test]
    fn powf() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let powered = tensor.powf(2.0)?;
        assert_eq!(powered.read()?, vec![1.0, 4.0, 9.0]);
        Ok(())
    }

    #[test]
    fn element_wise_pow() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let tensor2 = RowVector::<f32, 3>::new(&[2.0, 2.0, 2.0], &device)?;
        let powered = tensor1.pow(&tensor2)?;
        assert_eq!(powered.read()?, vec![1.0, 4.0, 9.0]);
        Ok(())
    }

    #[test]
    fn sinh() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[0.0, 1.0, -1.0], &device)?;
        let sinh = tensor.sinh()?;
        let sinh_vec = sinh.read()?;
        assert_relative_eq!(sinh_vec[0], 0.0);
        assert_relative_eq!(sinh_vec[1], 1.1752012);
        assert_relative_eq!(sinh_vec[2], -1.1752012);
        Ok(())
    }

    #[test]
    fn cosh() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[0.0, 1.0, -1.0], &device)?;
        let cosh = tensor.cosh()?;
        let cosh_vec = cosh.read()?;
        assert_relative_eq!(cosh_vec[0], 1.0);
        assert_relative_eq!(cosh_vec[1], 1.5430806);
        assert_relative_eq!(cosh_vec[2], 1.5430806);
        Ok(())
    }

    #[test]
    fn transpose() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let transposed = tensor.transpose()?;
        let trans_vec = transposed.read()?;
        assert_eq!(trans_vec, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn invalid_size() {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0]; // Wrong size (2 instead of 3)
        let _tensor = RowVector::<f32, 3>::new(&data, &device).unwrap();
    }

    #[test]
    fn multiple_operations_chain() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let result = tensor.mul_scalar(2.0)?.add_scalar(1.0)?.pow_scalar(2.0)?;
        assert_relative_eq_vec(result.read()?, vec![9.0, 25.0, 49.0]);
        Ok(())
    }

    #[test]
    fn different_dtypes() -> Result<()> {
        let device = Device::Cpu;

        // Test with f64
        let tensor_f64 = RowVector::<f64, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        assert_eq!(tensor_f64.0.dtype(), DType::F64);

        // Test with f32
        let tensor_f32 = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        assert_eq!(tensor_f32.0.dtype(), DType::F32);

        Ok(())
    }

    #[test]
    fn edge_cases() -> Result<()> {
        let device = Device::Cpu;

        // Test with very large numbers
        let large = RowVector::<f32, 3>::new(&[1e38, 1e38, 1e38], &device)?;
        let large_mul = large.mul_scalar(2.0)?;
        assert_relative_eq!(
            large_mul.read()?[0],
            200000000000000000000000000000000000000f32
        );

        // Test with very small numbers
        let small = RowVector::<f32, 3>::new(&[1e-38, 1e-38, 1e-38], &device)?;
        let small_div = small.div_scalar(2.0)?;
        assert!(small_div.read()?[0] != 0.0);

        // Test with zero division
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;
        let zero = RowVector::<f32, 3>::zeros(&device)?;
        let div_zero = tensor.div(&zero)?;
        assert!(div_zero.read()?[0].is_infinite());

        Ok(())
    }

    #[test]
    fn broadcasting_behavior() -> Result<()> {
        let device = Device::Cpu;
        let tensor = RowVector::<f32, 3>::new(&[1.0, 2.0, 3.0], &device)?;

        // Test broadcasting with scalar operations
        let scalar_add = tensor.add_scalar(1.0)?;
        let scalar_mul = tensor.mul_scalar(2.0)?;

        assert_eq!(scalar_add.0.shape().dims(), [1, 3]);
        assert_eq!(scalar_mul.0.shape().dims(), [1, 3]);
        assert_eq!(scalar_add.read()?, vec![2.0, 3.0, 4.0]);
        assert_eq!(scalar_mul.read()?, vec![2.0, 4.0, 6.0]);

        Ok(())
    }
}
