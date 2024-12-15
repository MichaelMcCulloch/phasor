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

#[cfg(test)]
mod test {
    use crate::*;
    use approx::assert_relative_eq;
    use candle_core::{Device, Result};
    use operands::test::assert_relative_eq_vec_vec;

    #[test]
    fn new_tensor2d() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Matrix::<f64, 2, 3>::new(&data, &device)?;
        assert_eq!(tensor.0.shape().dims(), [2, 3]);
        let values = tensor.read()?;
        assert_eq!(values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn zeros() -> Result<()> {
        let device = Device::Cpu;
        let zeros = Matrix::<f64, 2, 3>::zeros(&device)?;
        let values = zeros.read()?;
        assert_eq!(values, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
        Ok(())
    }

    #[test]
    fn ones() -> Result<()> {
        let device = Device::Cpu;
        let ones = Matrix::<f64, 2, 3>::ones(&device)?;
        let values = ones.read()?;
        assert_eq!(values, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        Ok(())
    }

    #[test]
    fn ones_neg() -> Result<()> {
        let device = Device::Cpu;
        let ones_neg = Matrix::<f64, 2, 3>::ones_neg(&device)?;
        let values = ones_neg.read()?;
        assert_eq!(values, vec![vec![-1.0, -1.0, -1.0], vec![-1.0, -1.0, -1.0]]);
        Ok(())
    }

    #[test]
    fn neg() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let neg = tensor.neg()?;
        let values = neg.read()?;
        assert_eq!(values, vec![vec![-1.0, -2.0], vec![-3.0, -4.0]]);
        Ok(())
    }

    #[test]
    fn abs() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[-1.0, -2.0, 3.0, -4.0], &device)?;
        let abs = tensor.abs()?;
        let values = abs.read()?;
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        Ok(())
    }
    #[test]
    fn add_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.add_scalar(2.0)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn sub_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.sub_scalar(1.0)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
        Ok(())
    }

    #[test]
    fn mul_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.mul_scalar(2.0)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn div_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[2.0, 4.0, 6.0, 8.0], &device)?;
        let result = tensor.div_scalar(2.0)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn pow_scalar() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f32, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.pow_scalar(2.0)?;
        let values = result.read()?;
        assert_relative_eq_vec_vec(values, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
        Ok(())
    }

    // Part 3: Element-wise Operations Tests

    #[test]
    fn add() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let tensor2 = Matrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &device)?;
        let result = tensor1.add(&tensor2)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn sub() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = Matrix::<f64, 2, 2>::new(&[2.0, 3.0, 4.0, 5.0], &device)?;
        let tensor2 = Matrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &device)?;
        let result = tensor1.sub(&tensor2)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn mul() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let tensor2 = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
        let result = tensor1.mul(&tensor2)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn div() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = Matrix::<f64, 2, 2>::new(&[2.0, 4.0, 6.0, 8.0], &device)?;
        let tensor2 = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
        let result = tensor1.div(&tensor2)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn element_wise_pow() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let tensor2 = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
        let result = tensor1.pow(&tensor2)?;
        let values = result.read()?;
        assert_relative_eq_vec_vec(values, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
        Ok(())
    }
    #[test]
    fn matmul() -> Result<()> {
        let device = Device::Cpu;
        let tensor1 = Matrix::<f64, 2, 3>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?;
        let tensor2 = Matrix::<f64, 3, 2>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?;
        let result = tensor1.matmul(&tensor2)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![22.0, 28.0], vec![49.0, 64.0]]);
        Ok(())
    }

    #[test]
    fn transpose() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 3>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?;
        let transposed = tensor.transpose()?;
        let values = transposed.read()?;
        assert_eq!(values, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        Ok(())
    }

    // Part 5: Transcendental Functions Tests

    #[test]
    fn exp() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, 0.5, 2.0], &device)?;
        let result = tensor.exp()?;
        let values = result.read()?;
        assert_relative_eq!(values[0][0], 1.0);
        assert_relative_eq!(values[0][1], 2.718281828459045);
        assert_relative_eq!(values[1][0], 1.6487212707001282);
        assert_relative_eq!(values[1][1], 7.38905609893065);
        Ok(())
    }

    #[test]
    fn log() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 4.0, 8.0], &device)?;
        let result = tensor.log()?;
        let values = result.read()?;
        assert_relative_eq!(values[0][0], 0.0);
        assert_relative_eq!(values[0][1], 0.6931471805599453);
        assert_relative_eq!(values[1][0], 1.3862943611198906);
        assert_relative_eq!(values[1][1], 2.0794415416798357);
        Ok(())
    }

    #[test]
    fn sin() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(
            &[
                0.0,
                std::f64::consts::PI / 2.0,
                std::f64::consts::PI,
                3.0 * std::f64::consts::PI / 2.0,
            ],
            &device,
        )?;
        let result = tensor.sin()?;
        let values = result.read()?;
        assert_relative_eq!(values[0][0], 0.0);
        assert_relative_eq!(values[0][1], 1.0);
        assert_relative_eq!(values[1][0], 0.0);
        assert_relative_eq!(values[1][1], -1.0);
        Ok(())
    }

    #[test]
    fn cos() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(
            &[
                0.0,
                std::f64::consts::PI / 2.0,
                std::f64::consts::PI,
                3.0 * std::f64::consts::PI / 2.0,
            ],
            &device,
        )?;
        let result = tensor.cos()?;
        let values = result.read()?;
        assert_relative_eq!(values[0][0], 1.0);
        assert_relative_eq!(values[0][1], 0.0);
        assert_relative_eq!(values[1][0], -1.0);
        assert_relative_eq!(values[1][1], 0.0);
        Ok(())
    }

    #[test]
    fn sinh() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, -1.0, 2.0], &device)?;
        let result = tensor.sinh()?;
        let values = result.read()?;
        assert_relative_eq!(values[0][0], 0.0);
        assert_relative_eq!(values[0][1], 1.1752011936438014);
        assert_relative_eq!(values[1][0], -1.1752011936438014);
        assert_relative_eq!(values[1][1], 3.626860407847019);
        Ok(())
    }

    #[test]
    fn cosh() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, -1.0, 2.0], &device)?;
        let result = tensor.cosh()?;
        let values = result.read()?;
        assert_relative_eq!(values[0][0], 1.0);
        assert_relative_eq!(values[0][1], 1.5430806348152437);
        assert_relative_eq!(values[1][0], 1.5430806348152437);
        assert_relative_eq!(values[1][1], 3.7621956910836314);
        Ok(())
    }

    #[test]
    fn tanh() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[0.0, 1.0, -1.0, 2.0], &device)?;
        let result = tensor.tanh()?;
        let values = result.read()?;
        assert_relative_eq!(values[0][0], 0.0);
        assert_relative_eq!(values[0][1], 0.7615941559557649);
        assert_relative_eq!(values[1][0], -0.7615941559557649);
        assert_relative_eq!(values[1][1], 0.9640275800758169);
        Ok(())
    }

    #[test]
    fn powf() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Matrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.powf(2.0)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
        Ok(())
    }

    #[test]
    fn where_cond() -> Result<()> {
        let device = Device::Cpu;
        let condition = Matrix::<u8, 2, 2>::new(&[1, 0, 1, 0], &device)?;
        let on_true = Matrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &device)?;
        let on_false = Matrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &device)?;
        let result = condition.where_cond(&on_true, &on_false)?;
        let values = result.read()?;
        assert_eq!(values, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
        Ok(())
    }
}
