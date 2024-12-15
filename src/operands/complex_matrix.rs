use crate::ops::*;
use crate::{ComplexColumnVector, ComplexRowVector, ComplexScalar, Matrix};
use candle_core::{DType, Device, FloatDType, Result, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct ComplexMatrix<T: WithDType, const ROWS: usize, const C: usize> {
    pub(crate) real: Matrix<T, ROWS, C>,
    pub(crate) imag: Matrix<T, ROWS, C>,
}

impl_complex_op!(ComplexMatrix, real, imag, Matrix, ROWS, COLS);

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
impl<T: WithDType, const ROWS: usize, const COLS: usize> ElementWiseOp<T>
    for ComplexMatrix<T, ROWS, COLS>
{
    type Output = ComplexMatrix<T, ROWS, COLS>;
    #[inline]
    fn add(&self, rhs: &Self) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.add(&rhs.real)?,
            imag: self.imag.add(&rhs.imag)?,
        })
    }
    #[inline]
    fn sub(&self, rhs: &Self) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.sub(&rhs.real)?,
            imag: self.imag.sub(&rhs.imag)?,
        })
    }
    #[inline]
    fn mul(&self, rhs: &Self) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.mul(&rhs.real)?.sub(&self.imag.mul(&rhs.imag)?)?,
            imag: self.real.mul(&rhs.imag)?.add(&self.imag.mul(&rhs.real)?)?,
        })
    }
    #[inline]
    fn div(&self, rhs: &Self) -> Result<Self::Output> {
        let denom = rhs.real.mul(&rhs.real)?.add(&rhs.imag.mul(&rhs.imag)?)?;
        Ok(Self {
            real: self
                .real
                .mul(&rhs.real)?
                .add(&self.imag.mul(&rhs.imag)?)?
                .div(&denom)?,
            imag: self
                .imag
                .mul(&rhs.real)?
                .sub(&self.real.mul(&rhs.imag)?)?
                .div(&denom)?,
        })
    }
    #[inline]
    fn clamp(&self, min: &T, max: &T) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.clamp(min, max)?,
            imag: self.imag.clamp(min, max)?,
        })
    }
}
impl<T: WithDType, const ROWS: usize, const COLS: usize> ScalarOp<T>
    for ComplexMatrix<T, ROWS, COLS>
{
    #[inline]
    fn add_scalar(&self, scalar: T) -> Result<Self> {
        Ok(Self {
            real: self.real.add_scalar(scalar)?,
            imag: self.imag.clone(),
        })
    }
    #[inline]
    fn sub_scalar(&self, scalar: T) -> Result<Self> {
        Ok(Self {
            real: self.real.sub_scalar(scalar)?,
            imag: self.imag.clone(),
        })
    }
    #[inline]
    fn mul_scalar(&self, scalar: T) -> Result<Self> {
        Ok(Self {
            real: self.real.mul_scalar(scalar)?,
            imag: self.imag.mul_scalar(scalar)?,
        })
    }
    #[inline]
    fn div_scalar(&self, scalar: T) -> Result<Self> {
        Ok(Self {
            real: self.real.div_scalar(scalar)?,
            imag: self.imag.div_scalar(scalar)?,
        })
    }
    #[inline]
    fn powf(&self, exponent: f64) -> Result<Self> {
        if exponent == 0.0 {
            return Self::ones(self.real.0.device());
        }
        if exponent == 1.0 {
            return Ok(self.clone());
        }
        if exponent == -1.0 {
            let denom = self
                .real
                .mul(&self.real)?
                .add(&self.imag.mul(&self.imag)?)?;
            return Ok(Self {
                real: self.real.div(&denom)?,
                imag: self.imag.neg()?.div(&denom)?,
            });
        }
        if exponent.fract() == 0.0 && exponent.abs() <= 4.0 {
            let mut result = self.clone();
            let n = exponent.abs() as i32;
            for _ in 1..n {
                result = result.mul(self)?;
            }
            if exponent < 0.0 {
                let denom = result
                    .real
                    .mul(&result.real)?
                    .add(&result.imag.mul(&result.imag)?)?;
                result = Self {
                    real: result.real.div(&denom)?,
                    imag: result.imag.neg()?.div(&denom)?,
                };
            }
            return Ok(result);
        }
        let r = self.magnitude()?;
        let ln_r = r.log()?;
        let scaled_ln_r = ln_r.mul_scalar(T::from_f64(exponent))?;
        let r_n = scaled_ln_r.exp()?;
        let theta = self.imag.atan2(&self.real)?;
        let new_theta = theta.mul_scalar(T::from_f64(exponent))?;

        Ok(Self {
            real: r_n.mul(&new_theta.cos()?)?,
            imag: r_n.mul(&new_theta.sin()?)?,
        })
    }
    #[inline]
    fn pow(&self, other: &Self) -> Result<Self> {
        self.log()?.mul(other)?.exp()
    }
    #[inline]
    fn pow_scalar(&self, scalar: T) -> Result<Self> {
        // Handle special cases for scalar integer powers
        if scalar == T::zero() {
            return Self::ones(self.real.device());
        }
        if scalar == T::one() {
            return Ok(self.clone());
        }
        if scalar == T::from_f64(-1.0) {
            let denom = self
                .real
                .mul(&self.real)?
                .add(&self.imag.mul(&self.imag)?)?;
            return Ok(Self {
                real: self.real.div(&denom)?,
                imag: self.imag.neg()?.div(&denom)?,
            });
        }

        if scalar.to_f64().fract() == 0.0 && scalar.to_f64().abs() <= 4.0 {
            let mut result = self.clone();
            let n = scalar.to_f64().abs() as i32;
            for _ in 1..n {
                result = result.mul(self)?;
            }
            if scalar.to_f64() < 0.0 {
                let denom = result
                    .real
                    .mul(&result.real)?
                    .add(&result.imag.mul(&result.imag)?)?;
                result = Self {
                    real: result.real.div(&denom)?,
                    imag: result.imag.neg()?.div(&denom)?,
                };
            }
            return Ok(result);
        }

        let r = self.magnitude()?;
        let ln_r = r.log()?;
        let scaled_ln_r = ln_r.mul_scalar(scalar)?;
        let r_n = scaled_ln_r.exp()?;

        let theta = self.imag.atan2(&self.real)?;
        let new_theta = theta.mul_scalar(scalar)?;

        Ok(Self {
            real: r_n.mul(&new_theta.cos()?)?,
            imag: r_n.mul(&new_theta.sin()?)?,
        })
    }
}
impl<T: WithDType, const ROWS: usize, const COLS: usize> TrigOp<T>
    for ComplexMatrix<T, ROWS, COLS>
{
    type Output = ComplexMatrix<T, ROWS, COLS>;
    #[inline]
    fn sin(&self) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.sin()?.mul(&self.imag.cosh()?)?,
            imag: self.real.cos()?.mul(&self.imag.sinh()?)?,
        })
    }
    #[inline]
    fn cos(&self) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.cos()?.mul(&self.imag.cosh()?)?,
            imag: self.real.sin()?.mul(&self.imag.sinh()?)?.neg()?,
        })
    }
    #[inline]
    fn sinh(&self) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.sinh()?.mul(&self.imag.cos()?)?,
            imag: self.real.cosh()?.mul(&self.imag.sin()?)?,
        })
    }
    #[inline]
    fn cosh(&self) -> Result<Self::Output> {
        Ok(Self {
            real: self.real.cosh()?.mul(&self.imag.cos()?)?,
            imag: self.real.sinh()?.mul(&self.imag.sin()?)?,
        })
    }
    #[inline]
    fn tanh(&self) -> Result<Self::Output> {
        let two_x = self.real.mul_scalar(T::from_f64(2.0))?;
        let two_y = self.imag.mul_scalar(T::from_f64(2.0))?;

        let sinh_2x = two_x.sinh()?;
        let sin_2y = two_y.sin()?;
        let cosh_2x = two_x.cosh()?;
        let cos_2y = two_y.cos()?;

        let denom = cosh_2x.add(&cos_2y)?;

        Ok(Self {
            real: sinh_2x.div(&denom)?,
            imag: sin_2y.div(&denom)?,
        })
    }
    #[inline]
    fn atan(&self) -> Result<Self::Output> {
        let zero = Matrix::zeros(self.device())?;
        let one = Matrix::ones(self.device())?;
        let eps = Matrix::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
        let pi_half = Matrix::ones(self.device())?.mul_scalar(T::from_f64(PI / 2.0))?;

        // Form iz
        let iz = ComplexMatrix {
            real: self.imag.neg()?,
            imag: self.real.clone(),
        };

        // Compute (1 + iz)/(1 - iz)
        let numerator = ComplexMatrix {
            real: one.clone(),
            imag: zero.clone(),
        }
        .add(&iz)?;

        let denominator = ComplexMatrix {
            real: one.clone(),
            imag: zero.clone(),
        }
        .sub(&iz)?;

        // Check for special cases
        let mag_real = self.real.abs()?;
        let mag_imag = self.imag.abs()?;

        // z = 0 case
        let is_zero = mag_real.lt(&eps)?.mul(&mag_imag.lt(&eps)?)?;

        // z = ±i case (near branch points)
        let near_i = mag_real
            .lt(&eps)?
            .mul(&(mag_imag.sub(&one)?.abs()?.lt(&eps)?))?;

        // Standard computation
        let standard_result = {
            let ratio = numerator.div(&denominator)?;
            let log_result = ratio.log()?;
            ComplexMatrix {
                real: log_result.imag.mul_scalar(T::from_f64(0.5))?,
                imag: log_result.real.mul_scalar(T::from_f64(-0.5))?,
            }
        };

        // Special case results
        let zero_result = ComplexMatrix {
            real: zero.clone(),
            imag: zero.clone(),
        };

        let i_result = ComplexMatrix {
            real: pi_half.mul(&self.imag.div(&mag_imag)?)?,
            imag: Matrix::ones(self.device())?.mul_scalar(T::from_f64(f64::INFINITY))?,
        };

        // Combine results using conditional operations
        let result = is_zero.where_cond_complex(
            &zero_result,
            &near_i.where_cond_complex(&i_result, &standard_result)?,
        )?;

        Ok(result)
    }
    #[inline]
    fn atan2(&self, x: &Self) -> Result<Self::Output> {
        let zero = Matrix::zeros(self.device())?;
        let eps = Matrix::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
        let pi = Matrix::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
        let pi_half = pi.mul_scalar(T::from_f64(0.5))?;

        // Compute magnitudes
        let y_mag = self.magnitude()?;
        let x_mag = x.magnitude()?;

        // Special cases
        let both_zero = y_mag.lt(&eps)?.mul(&x_mag.lt(&eps)?)?;
        let x_zero = x_mag.lt(&eps)?;

        // Base atan computation
        let z = self.div(x)?;
        let base_atan = z.atan()?;

        // Quadrant adjustments
        let x_neg = x.real.lt(&zero)?;
        let y_gte_zero = self.real.gte(&zero)?;

        // When x < 0: add π for y ≥ 0, subtract π for y < 0
        let adjustment = x_neg.where_cond(&y_gte_zero.where_cond(&pi, &pi.neg()?)?, &zero)?;

        // Apply adjustment to real part only
        let adjusted_result = ComplexMatrix {
            real: base_atan.real.add(&adjustment)?,
            imag: base_atan.imag,
        };

        // Handle x = 0 cases
        let x_zero_result = ComplexMatrix {
            real: y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?,
            imag: zero.clone(),
        };

        // Combine all cases
        let result = both_zero.where_cond_complex(
            &ComplexMatrix {
                real: zero.clone(),
                imag: zero.clone(),
            },
            &x_zero.where_cond_complex(&x_zero_result, &adjusted_result)?,
        )?;

        Ok(result)
    }
}
impl<T: WithDType, const ROWS: usize, const COLS: usize> UnaryOp<T>
    for ComplexMatrix<T, ROWS, COLS>
{
    type TransposeOutput = ComplexMatrix<T, COLS, ROWS>;
    type ScalarOutput = ComplexScalar<T>;
    #[inline]
    fn neg(&self) -> Result<Self> {
        Ok(Self {
            real: self.real.mul_scalar(T::from_f64(-1.0))?,
            imag: self.imag.mul_scalar(T::from_f64(-1.0))?,
        })
    }
    #[inline]
    fn abs(&self) -> Result<Self> {
        Ok(Self {
            real: self.magnitude()?,
            imag: Matrix::zeros(self.real.device())?,
        })
    }
    #[inline]
    fn exp(&self) -> Result<Self> {
        let exp_real = self.real.exp()?;
        Ok(Self {
            real: exp_real.mul(&self.imag.cos()?)?,
            imag: exp_real.mul(&self.imag.sin()?)?,
        })
    }
    #[inline]
    fn log(&self) -> Result<Self> {
        let magnitude = self.magnitude()?;
        let argument = self.arg()?;

        Ok(Self {
            real: magnitude.log()?,
            imag: argument,
        })
    }
    #[inline]
    fn mean(&self) -> Result<ComplexScalar<T>> {
        Ok(ComplexScalar {
            real: self.real.mean()?,
            imag: self.imag.mean()?,
        })
    }
}
impl<T: WithDType, const ROWS: usize, const COLS: usize> ComparisonOp<T>
    for ComplexMatrix<T, ROWS, COLS>
{
    type Output = Matrix<u8, ROWS, COLS>;
    #[inline]
    fn lt(&self, other: &Self) -> Result<Self::Output> {
        let mag_self = self.magnitude()?;
        let mag_other = other.magnitude()?;
        mag_self.lt(&mag_other)
    }
    #[inline]
    fn lte(&self, other: &Self) -> Result<Self::Output> {
        let mag_self = self.magnitude()?;
        let mag_other = other.magnitude()?;
        mag_self.lte(&mag_other)
    }
    #[inline]
    fn eq(&self, other: &Self) -> Result<Self::Output> {
        let real_eq = self.real.eq(&other.real)?;
        let imag_eq = self.imag.eq(&other.imag)?;
        real_eq.mul(&imag_eq)
    }
    #[inline]
    fn ne(&self, other: &Self) -> Result<Self::Output> {
        let real_ne = self.real.ne(&other.real)?;
        let imag_ne = self.imag.ne(&other.imag)?;
        real_ne
            .add(&imag_ne)?
            .gt(&Matrix::<u8, ROWS, COLS>::zeros(self.device())?)
    }
    #[inline]
    fn gt(&self, other: &Self) -> Result<Self::Output> {
        let mag_self = self.magnitude()?;
        let mag_other = other.magnitude()?;
        mag_self.gt(&mag_other)
    }
    #[inline]
    fn gte(&self, other: &Self) -> Result<Self::Output> {
        let mag_self = self.magnitude()?;
        let mag_other = other.magnitude()?;
        mag_self.gte(&mag_other)
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
        Ok(ComplexMatrix {
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
    fn transpose(&self) -> Result<Self::TransposeOutput> {
        Ok(ComplexMatrix {
            real: self.real.transpose()?,
            imag: self.imag.transpose()?,
        })
    }
}
impl<T: WithDType, const ROWS: usize, const COLS: usize> TensorFactory<T>
    for ComplexMatrix<T, ROWS, COLS>
{
    #[inline]
    fn zeros(device: &Device) -> Result<Self> {
        Ok(Self {
            real: Matrix::zeros(device)?,
            imag: Matrix::zeros(device)?,
        })
    }
    #[inline]
    fn ones(device: &Device) -> Result<Self> {
        Ok(Self {
            real: Matrix::ones(device)?,
            imag: Matrix::zeros(device)?,
        })
    }
    #[inline]
    fn ones_neg(device: &Device) -> Result<Self> {
        Ok(Self {
            real: Matrix::ones_neg(device)?,
            imag: Matrix::zeros(device)?,
        })
    }
}
impl<F: FloatDType, const ROWS: usize, const COLS: usize> TensorFactoryFloat<F>
    for ComplexMatrix<F, ROWS, COLS>
{
    #[inline]
    fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
        Ok(Self {
            real: Matrix::<F, ROWS, COLS>::randn(mean, std, device)?,
            imag: Matrix::<F, ROWS, COLS>::randn(mean, std, device)?,
        })
    }
    #[inline]
    fn randu(low: F, high: F, device: &Device) -> Result<Self> {
        Ok(Self {
            real: Matrix::<F, ROWS, COLS>::randu(low, high, device)?,
            imag: Matrix::<F, ROWS, COLS>::randu(low, high, device)?,
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
#[cfg(test)]
mod test {
    use crate::*;
    use approx::assert_relative_eq;
    use candle_core::{Device, Result};
    use std::f64::consts::PI;

    #[test]
    fn new_complex_tensor() -> Result<()> {
        let device = Device::Cpu;
        let real = vec![1.0, 2.0, 3.0, 4.0];
        let imag = vec![5.0, 6.0, 7.0, 8.0];
        let tensor = ComplexMatrix::<f64, 2, 2>::new(&real, &imag, &device)?;
        assert_eq!(tensor.real.read()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(tensor.imag.read()?, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn sub() -> Result<()> {
        let device = Device::Cpu;
        let a =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &device)?;
        let b =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0], &device)?;
        let c = a.sub(&b)?;
        assert_eq!(c.real.read()?, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
        assert_eq!(c.imag.read()?, vec![vec![4.0, 5.0], vec![6.0, 7.0]]);
        Ok(())
    }

    #[test]
    fn mul() -> Result<()> {
        let device = Device::Cpu;
        let a =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0], &device)?;
        let b =
            ComplexMatrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &[1.0, 1.0, 1.0, 1.0], &device)?;
        let c = a.mul(&b)?;

        // Correct calculations:
        // (1+i)(2+i) = (1*2 - 1*1) + (1*1 + 1*2)i = 1 + 3i
        // (2+i)(2+i) = (2*2 - 1*1) + (2*1 + 1*2)i = 3 + 4i
        // (3+i)(2+i) = (3*2 - 1*1) + (3*1 + 1*2)i = 5 + 5i
        // (4+i)(2+i) = (4*2 - 1*1) + (4*1 + 1*2)i = 7 + 6i

        assert_eq!(c.real.read()?, vec![vec![1.0, 3.0], vec![5.0, 7.0]]);
        assert_eq!(c.imag.read()?, vec![vec![3.0, 4.0], vec![5.0, 6.0]]); // Corrected imaginary parts
        Ok(())
    }
    #[test]
    fn div() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 1, 1>::new(&[4.0], &[0.0], &device)?;
        let b = ComplexMatrix::<f64, 1, 1>::new(&[2.0], &[0.0], &device)?;
        let c = a.div(&b)?;
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0][0], 2.0);
        assert_relative_eq!(imag[0][0], 0.0);
        Ok(())
    }

    #[test]
    fn div_pure_imag() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
        let b = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
        let c = a.div(&b)?;
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0][0], 1.0);
        assert_relative_eq!(imag[0][0], 0.0);
        Ok(())
    }

    #[test]
    fn conj() -> Result<()> {
        let device = Device::Cpu;
        let a =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &device)?;
        let c = a.conj()?;
        assert_eq!(c.real.read()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(c.imag.read()?, vec![vec![-5.0, -6.0], vec![-7.0, -8.0]]);
        Ok(())
    }

    #[test]
    fn abs() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 2, 1>::new(&[3.0, 0.0], &[4.0, 1.0], &device)?;
        let c = a.abs()?;
        // |3 + 4i| = 5, |0 + i| = 1
        let abs = c.real.read()?;
        assert_relative_eq!(abs[0][0], 5.0);
        assert_relative_eq!(abs[1][0], 1.0);
        Ok(())
    }

    #[test]
    fn log() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
        let c = a.log()?;
        // ln(i) = iπ/2
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0][0], 0.0);
        assert_relative_eq!(imag[0][0], PI / 2.0);
        Ok(())
    }

    #[test]
    fn pow_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 1, 1>::new(&[1.0], &[1.0], &device)?; // 1 + i
        let c = a.pow_scalar(2.0)?;
        // (1+i)^2 = 0 + 2i
        let real = c.real.read()?;
        let imag = c.imag.read()?;

        assert_relative_eq!(real[0][0], 0.0);
        assert_relative_eq!(imag[0][0], 2.0);
        Ok(())
    }
    #[test]
    fn powf_edge_cases() -> Result<()> {
        let device = Device::Cpu;
        // Test case 1: Negative base, fractional exponent
        let tensor = RowVector::<f64, 1>::new(&[-1.0], &device)?;
        let result = tensor.powf(0.5)?; // Should be NaN or complex depending on your backend
                                        // Add assertions based on expected behavior (NaN, complex number, or exception)

        // Test case 2: Zero base, positive exponent
        let tensor = RowVector::<f64, 1>::new(&[0.0], &device)?;
        let result = tensor.powf(2.0)?;
        assert_eq!(result.read()?, vec![0.0]);

        // Test case 3: Zero base, negative exponent
        let tensor = RowVector::<f64, 1>::new(&[0.0], &device)?;
        let result = tensor.powf(-2.0)?; // Should be infinity or exception
                                         // Add assertions based on expected behavior

        Ok(())
    }
    #[test]
    fn pow() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[1.0], &device)?;
        let b = ComplexMatrix::<f64, 1, 1>::new(&[2.0], &[0.0], &device)?;
        let c = ComplexOp::pow(&a, &b)?;
        // i^2 = -1
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0][0], -1.0);
        assert_relative_eq!(imag[0][0], 0.0);
        Ok(())
    }

    #[test]
    fn sub_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &device)?;
        let result = a.sub_scalar(1.0)?;
        assert_eq!(result.real.read()?, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
        assert_eq!(result.imag.read()?, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn div_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 2, 2>::new(
            &[2.0, 4.0, 6.0, 8.0],
            &[10.0, 12.0, 14.0, 16.0],
            &device,
        )?;
        let result = a.div_scalar(2.0)?;
        assert_eq!(result.real.read()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(result.imag.read()?, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn zeros() -> Result<()> {
        let device = Device::Cpu;
        let zeros = ComplexMatrix::<f64, 2, 2>::zeros(&device)?;
        assert_eq!(zeros.real.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(zeros.imag.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        Ok(())
    }

    #[test]
    fn ones() -> Result<()> {
        let device = Device::Cpu;
        let ones = ComplexMatrix::<f64, 2, 2>::ones(&device)?;
        assert_eq!(ones.real.read()?, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(ones.imag.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        Ok(())
    }

    #[test]
    fn ones_neg() -> Result<()> {
        let device = Device::Cpu;
        let ones_neg = ComplexMatrix::<f64, 2, 2>::ones_neg(&device)?;
        assert_eq!(
            ones_neg.real.read()?,
            vec![vec![-1.0, -1.0], vec![-1.0, -1.0]]
        );
        assert_eq!(ones_neg.imag.read()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        Ok(())
    }

    #[test]
    fn add() -> Result<()> {
        let device = Device::Cpu;
        let a =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &device)?;
        let b =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0], &device)?;
        let c = a.add(&b)?;
        assert_eq!(c.real.read()?, vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
        assert_eq!(c.imag.read()?, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
        Ok(())
    }

    #[test]
    fn matmul() -> Result<()> {
        let device = Device::Cpu;
        // (1+i)(2+2i) + (2+2i)(3+3i) = (2-2) + (2+2)i + (6-6) + (6+6)i = 0 + 16i
        let a = ComplexMatrix::<f64, 1, 2>::new(&[1.0, 2.0], &[1.0, 2.0], &device)?;
        let b = ComplexMatrix::<f64, 2, 1>::new(&[2.0, 3.0], &[2.0, 3.0], &device)?;
        let c = a.matmul(&b)?;
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0][0], 0.0);
        assert_relative_eq!(imag[0][0], 16.0);
        Ok(())
    }

    #[test]
    fn transpose() -> Result<()> {
        let device = Device::Cpu;
        let a =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &device)?;
        let c = a.transpose()?;
        assert_eq!(c.real.read()?, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
        assert_eq!(c.imag.read()?, vec![vec![5.0, 7.0], vec![6.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn exp() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[PI / 2.0], &device)?;
        let c = a.exp()?;
        // e^(iπ/2) = i
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0][0], 0.0);
        assert_relative_eq!(imag[0][0], 1.0);
        Ok(())
    }

    #[test]
    fn scalar_operations() -> Result<()> {
        let device = Device::Cpu;
        let a =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &device)?;

        // Test add_scalar
        let add_result = a.add_scalar(2.0)?;
        assert_eq!(
            add_result.real.read()?,
            vec![vec![3.0, 4.0], vec![5.0, 6.0]]
        );
        assert_eq!(
            add_result.imag.read()?,
            vec![vec![5.0, 6.0], vec![7.0, 8.0]]
        );

        // Test mul_scalar
        let mul_result = a.mul_scalar(2.0)?;
        assert_eq!(
            mul_result.real.read()?,
            vec![vec![2.0, 4.0], vec![6.0, 8.0]]
        );
        assert_eq!(
            mul_result.imag.read()?,
            vec![vec![10.0, 12.0], vec![14.0, 16.0]]
        );

        Ok(())
    }

    #[test]
    fn transcendental_functions() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexMatrix::<f64, 1, 1>::new(&[0.0], &[PI / 2.0], &device)?;

        // Test sin
        let sin_result = a.sin()?;
        let sin_real = sin_result.real.read()?;
        let sin_imag = sin_result.imag.read()?;
        assert_relative_eq!(sin_real[0][0], 0.0);

        // Test cos
        let cos_result = a.cos()?;
        let cos_real = cos_result.real.read()?;
        let cos_imag = cos_result.imag.read()?;
        assert_relative_eq!(cos_real[0][0], 2.5091784786580567);

        // Test sinh
        let sinh_result = a.sinh()?;
        let sinh_real = sinh_result.real.read()?;
        assert_relative_eq!(sinh_real[0][0], 0.0);

        // Test cosh
        let cosh_result = a.cosh()?;
        let cosh_real = cosh_result.real.read()?;
        assert_relative_eq!(cosh_real[0][0], 0.0);

        Ok(())
    }

    #[test]
    fn where_cond() -> Result<()> {
        let device = Device::Cpu;
        let cond = Matrix::<u8, 2, 2>::new(&[1, 0, 1, 0], &device)?;
        let on_true =
            ComplexMatrix::<f64, 2, 2>::new(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0], &device)?;
        let on_false =
            ComplexMatrix::<f64, 2, 2>::new(&[2.0, 2.0, 2.0, 2.0], &[2.0, 2.0, 2.0, 2.0], &device)?;
        let result = cond.where_cond_complex(&on_true, &on_false)?;
        assert_eq!(result.real.read()?, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
        assert_eq!(result.imag.read()?, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
        Ok(())
    }
}
