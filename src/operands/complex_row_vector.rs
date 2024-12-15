use crate::ops::*;
use crate::{ComplexColumnVector, ComplexMatrix, ComplexScalar, RowVector, Scalar};
use candle_core::{DType, Device, FloatDType, Result, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct ComplexRowVector<T: WithDType, const ROWS: usize> {
    pub(crate) real: RowVector<T, ROWS>,
    pub(crate) imag: RowVector<T, ROWS>,
}

// In src/operands/complex_row_vector.rs
impl_complex_op!(ComplexRowVector, real, imag, RowVector, ROWS);

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
impl<T: WithDType, const COLS: usize> ElementWiseOp<T> for ComplexRowVector<T, COLS> {
    type Output = ComplexRowVector<T, COLS>;
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
impl<T: WithDType, const COLS: usize> ScalarOp<T> for ComplexRowVector<T, COLS> {
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
    fn pow_scalar(&self, scalar: T) -> Result<Self> {
        let r = self.magnitude()?;
        let theta = self.imag.atan2(&self.real)?; // Use atan2 for correct argument
        let r_pow = r.pow_scalar(scalar)?;
        let new_theta = theta.mul_scalar(scalar)?;
        Ok(Self {
            real: r_pow.mul(&new_theta.cos()?)?,
            imag: r_pow.mul(&new_theta.sin()?)?,
        })
    }
    #[inline]
    fn powf(&self, exponent: f64) -> Result<Self> {
        let r = self.magnitude()?;
        let theta = self.imag.atan2(&self.real)?;
        let r_pow = r.powf(exponent)?;
        let new_theta = theta.mul_scalar(T::from_f64(exponent))?;
        Ok(Self {
            real: r_pow.mul(&new_theta.cos()?)?,
            imag: r_pow.mul(&new_theta.sin()?)?,
        })
    }
    #[inline]
    fn pow(&self, other: &Self) -> Result<Self> {
        self.log()?.mul(other)?.exp()
    }
}
impl<T: WithDType, const COLS: usize> TrigOp<T> for ComplexRowVector<T, COLS> {
    type Output = ComplexRowVector<T, COLS>;
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
        let zero = RowVector::zeros(self.device())?;
        let one = RowVector::ones(self.device())?;
        let eps = RowVector::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
        let pi_half = RowVector::ones(self.device())?.mul_scalar(T::from_f64(PI / 2.0))?;

        // Form iz
        let iz = ComplexRowVector {
            real: self.imag.neg()?,
            imag: self.real.clone(),
        };

        // Compute (1 + iz)/(1 - iz)
        let numerator = ComplexRowVector {
            real: one.clone(),
            imag: zero.clone(),
        }
        .add(&iz)?;

        let denominator = ComplexRowVector {
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
            ComplexRowVector {
                real: log_result.imag.mul_scalar(T::from_f64(0.5))?,
                imag: log_result.real.mul_scalar(T::from_f64(-0.5))?,
            }
        };

        // Special case results
        let zero_result = ComplexRowVector {
            real: zero.clone(),
            imag: zero.clone(),
        };

        let i_result = ComplexRowVector {
            real: pi_half.mul(&self.imag.div(&mag_imag)?)?,
            imag: RowVector::ones(self.device())?.mul_scalar(T::from_f64(f64::INFINITY))?,
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
        let zero = RowVector::zeros(self.device())?;
        let eps = RowVector::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
        let pi = RowVector::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
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
        let adjusted_result = ComplexRowVector {
            real: base_atan.real.add(&adjustment)?,
            imag: base_atan.imag,
        };

        // Handle x = 0 cases
        let x_zero_result = ComplexRowVector {
            real: y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?,
            imag: zero.clone(),
        };

        // Combine all cases
        let result = both_zero.where_cond_complex(
            &ComplexRowVector {
                real: zero.clone(),
                imag: zero.clone(),
            },
            &x_zero.where_cond_complex(&x_zero_result, &adjusted_result)?,
        )?;

        Ok(result)
    }
}
impl<T: WithDType, const COLS: usize> UnaryOp<T> for ComplexRowVector<T, COLS> {
    type TransposeOutput = ComplexMatrix<T, 1, COLS>;
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
            imag: RowVector::zeros(self.real.device())?,
        })
    }
    #[inline]
    fn exp(&self) -> Result<Self> {
        let exp_real = self.real.exp()?;
        let cos_imag = self.imag.cos()?;
        let sin_imag = self.imag.sin()?;
        Ok(Self {
            real: exp_real.mul(&cos_imag)?,
            imag: exp_real.mul(&sin_imag)?,
        })
    }
    #[inline]
    fn log(&self) -> Result<Self> {
        let abs = self.magnitude()?;
        let zero = RowVector::<T, COLS>::zeros(self.real.0.device())?;
        let x_lt_zero = self.real.lt(&zero)?;
        let y_gte_zero = self.imag.gte(&zero)?;
        let ratio = self.imag.div(&self.real)?;
        let base_angle = ratio.tanh()?;
        let pi = RowVector::<T, COLS>::ones(self.real.0.device())?.mul_scalar(T::from_f64(PI))?;
        let pi_neg = pi.neg()?;
        let adjustment = x_lt_zero.where_cond(&y_gte_zero.where_cond(&pi, &pi_neg)?, &zero)?;
        let x_is_zero = self
            .real
            .abs()?
            .lt(&RowVector::<T, COLS>::ones(self.real.0.device())?
                .mul_scalar(T::from_f64(1e-10))?)?;
        let half_pi = pi.mul_scalar(T::from_f64(0.5))?;
        let neg_half_pi = half_pi.neg()?;
        let arg = x_is_zero.where_cond(
            &y_gte_zero.where_cond(&half_pi, &neg_half_pi)?,
            &base_angle.add(&adjustment)?,
        )?;

        Ok(Self {
            real: abs.log()?,
            imag: arg,
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
impl<T: WithDType, const COLS: usize> ComparisonOp<T> for ComplexRowVector<T, COLS> {
    type Output = RowVector<u8, COLS>;
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
            .gt(&RowVector::<u8, COLS>::zeros(self.device())?)
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
        let mag_other: RowVector<T, COLS> = other.magnitude()?;
        mag_self.gte(&mag_other)
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
impl<T: WithDType, const COLS: usize> TensorFactory<T> for ComplexRowVector<T, COLS> {
    #[inline]
    fn zeros(device: &Device) -> Result<Self> {
        Ok(Self {
            real: RowVector::zeros(device)?,
            imag: RowVector::zeros(device)?,
        })
    }
    #[inline]
    fn ones(device: &Device) -> Result<Self> {
        Ok(Self {
            real: RowVector::ones(device)?,
            imag: RowVector::zeros(device)?,
        })
    }
    #[inline]
    fn ones_neg(device: &Device) -> Result<Self> {
        Ok(Self {
            real: RowVector::ones_neg(device)?,
            imag: RowVector::zeros(device)?,
        })
    }
}
impl<F: FloatDType, const COLS: usize> TensorFactoryFloat<F> for ComplexRowVector<F, COLS> {
    #[inline]
    fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
        Ok(Self {
            real: RowVector::<F, COLS>::randn(mean, std, device)?,
            imag: RowVector::<F, COLS>::randn(mean, std, device)?,
        })
    }
    #[inline]
    fn randu(low: F, high: F, device: &Device) -> Result<Self> {
        Ok(Self {
            real: RowVector::<F, COLS>::randu(low, high, device)?,
            imag: RowVector::<F, COLS>::randu(low, high, device)?,
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
#[cfg(test)]
mod test {
    use crate::*;
    use approx::assert_relative_eq;
    use candle_core::{Device, Result};
    use operands::test::assert_relative_eq_vec;
    use std::f64::consts::PI;

    #[test]
    fn new_complex_tensor() -> Result<()> {
        let device = Device::Cpu;
        let real = vec![1.0, 2.0, 3.0];
        let imag = vec![4.0, 5.0, 6.0];
        let tensor = ComplexRowVector::<f64, 3>::new(&real, &imag, &device)?;
        assert_eq!(tensor.real.read()?, real);
        assert_eq!(tensor.imag.read()?, imag);
        Ok(())
    }

    #[test]
    fn zeros() -> Result<()> {
        let device = Device::Cpu;
        let zeros = ComplexRowVector::<f64, 3>::zeros(&device)?;
        assert_eq!(zeros.real.read()?, vec![0.0, 0.0, 0.0]);
        assert_eq!(zeros.imag.read()?, vec![0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn ones() -> Result<()> {
        let device = Device::Cpu;
        let ones = ComplexRowVector::<f64, 3>::ones(&device)?;
        assert_eq!(ones.real.read()?, vec![1.0, 1.0, 1.0]);
        assert_eq!(ones.imag.read()?, vec![0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn ones_neg() -> Result<()> {
        let device = Device::Cpu;
        let ones_neg = ComplexRowVector::<f64, 3>::ones_neg(&device)?;
        assert_eq!(ones_neg.real.read()?, vec![-1.0, -1.0, -1.0]);
        assert_eq!(ones_neg.imag.read()?, vec![0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn add() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;
        let b = ComplexRowVector::<f64, 2>::new(&[5.0, 6.0], &[7.0, 8.0], &device)?;
        let c = a.add(&b)?;
        assert_eq!(c.real.read()?, vec![6.0, 8.0]);
        assert_eq!(c.imag.read()?, vec![10.0, 12.0]);
        Ok(())
    }

    #[test]
    fn sub() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;
        let b = ComplexRowVector::<f64, 2>::new(&[5.0, 6.0], &[7.0, 8.0], &device)?;
        let c = a.sub(&b)?;
        assert_eq!(c.real.read()?, vec![-4.0, -4.0]);
        assert_eq!(c.imag.read()?, vec![-4.0, -4.0]);
        Ok(())
    }

    #[test]
    fn mul() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[1.0, 1.0], &device)?;
        let b = ComplexRowVector::<f64, 2>::new(&[2.0, 3.0], &[1.0, 1.0], &device)?;
        let c = a.mul(&b)?;
        // (1 + i)(2 + i) = (2 - 1) + (2 + 1)i = 1 + 3i
        // (2 + i)(3 + i) = (6 - 1) + (3 + 2)i = 5 + 5i
        assert_relative_eq_vec(c.real.read()?, vec![1.0, 5.0]);
        assert_relative_eq_vec(c.imag.read()?, vec![3.0, 5.0]);
        Ok(())
    }
    #[test]
    fn div_pure_real() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 1>::new(&[4.0], &[0.0], &device)?;
        let b = ComplexRowVector::<f64, 1>::new(&[2.0], &[0.0], &device)?;
        let c = a.div(&b)?;
        // (4 + 2i)/(2 + i) = (10 + 0i)/5 = 2 + 0i
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0], 2.0);
        assert_relative_eq!(imag[0], 0.0);
        Ok(())
    }
    #[test]
    fn div_pure_imag() -> Result<()> {
        // Case 1: i/i = 1
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[1.0], &device)?;
        let b = ComplexRowVector::<f64, 1>::new(&[0.0], &[1.0], &device)?;
        let c = a.div(&b)?;
        assert_relative_eq!(c.real.read()?[0], 1.0);
        assert_relative_eq!(c.imag.read()?[0], 0.0);
        Ok(())
    }
    #[test]
    fn div_unit_circle() -> Result<()> {
        // Case 2: (1 + i)/(1 - i) = 0 + i
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 1>::new(&[1.0], &[1.0], &device)?;
        let b = ComplexRowVector::<f64, 1>::new(&[1.0], &[-1.0], &device)?;
        let c = a.div(&b)?;
        assert_relative_eq!(c.real.read()?[0], 0.0);
        assert_relative_eq!(c.imag.read()?[0], 1.0);
        Ok(())
    }
    #[test]
    fn div_psychotic() -> Result<()> {
        // Case 3: (3 + 4i)/(2 + 2i) = 1.75 + 0.25i
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 1>::new(&[3.0], &[4.0], &device)?;
        let b = ComplexRowVector::<f64, 1>::new(&[2.0], &[2.0], &device)?;
        let c = a.div(&b)?;
        assert_relative_eq!(c.real.read()?[0], 1.75);
        assert_relative_eq!(c.imag.read()?[0], 0.25);

        Ok(())
    }

    #[test]
    fn exp() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[PI / 2.0], &device)?;
        let c = a.exp()?;
        // e^(iπ/2) = i
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0], 0.0);
        assert_relative_eq!(imag[0], 1.0);
        Ok(())
    }

    #[test]
    fn log() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[1.0], &device)?;
        let c = a.log()?;
        // ln(i) = iπ/2
        let real = c.real.read()?;
        let imag = c.imag.read()?;
        assert_relative_eq!(real[0], 0.0);
        assert_relative_eq!(imag[0], PI / 2.0);
        Ok(())
    }

    #[test]
    fn conj() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;
        let c = a.conj()?;
        assert_eq!(c.real.read()?, vec![1.0, 2.0]);
        assert_eq!(c.imag.read()?, vec![-3.0, -4.0]);
        Ok(())
    }

    #[test]
    fn abs() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 2>::new(&[3.0, 0.0], &[4.0, 1.0], &device)?;
        let c = UnaryOp::abs(&a)?;
        // |3 + 4i| = 5, |0 + i| = 1
        assert_eq!(c.real()?.read()?, vec![5.0, 1.0]); // Correct: Accessing c.real
        assert_eq!(c.imaginary()?.read()?, vec![0.0, 0.0]); // Correct: Verifying imaginary part is zero
        Ok(())
    }

    #[test]
    fn where_cond() -> Result<()> {
        let device = Device::Cpu;
        let cond = RowVector::<u8, 2>::new(&[1, 0], &device)?;
        let on_true = ComplexRowVector::<f64, 2>::new(&[1.0, 1.0], &[1.0, 1.0], &device)?;
        let on_false = ComplexRowVector::<f64, 2>::new(&[2.0, 2.0], &[2.0, 2.0], &device)?;
        let result = cond.where_cond_complex(&on_true, &on_false)?;
        assert_eq!(result.real.read()?, vec![1.0, 2.0]);
        assert_eq!(result.imag.read()?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn scalar_operations() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 2>::new(&[1.0, 2.0], &[3.0, 4.0], &device)?;

        // Test add_scalar
        let add_result = a.add_scalar(2.0)?;
        assert_eq!(add_result.real.read()?, vec![3.0, 4.0]);
        assert_eq!(add_result.imag.read()?, vec![3.0, 4.0]);

        // Test mul_scalar
        let mul_result = a.mul_scalar(2.0)?;
        assert_eq!(mul_result.real.read()?, vec![2.0, 4.0]);
        assert_eq!(mul_result.imag.read()?, vec![6.0, 8.0]);

        // Test div_scalar
        let div_result = a.div_scalar(2.0)?;
        assert_eq!(div_result.real.read()?, vec![0.5, 1.0]);
        assert_eq!(div_result.imag.read()?, vec![1.5, 2.0]);

        Ok(())
    }

    #[test]
    fn transcendental_functions() -> Result<()> {
        let device = Device::Cpu;
        let a = ComplexRowVector::<f64, 1>::new(&[0.0], &[PI / 2.0], &device)?;

        // Test sin
        let sin_result = a.sin()?;
        let sin_real = sin_result.real.read()?;
        let sin_imag = sin_result.imag.read()?;
        assert_relative_eq!(sin_real[0], 0.0);

        // Test cos
        let cos_result = a.cos()?;
        let cos_real = cos_result.real.read()?;
        let cos_imag = cos_result.imag.read()?;
        assert_relative_eq!(cos_real[0], 2.5091784786580567);

        // Test sinh
        let sinh_result = a.sinh()?;
        let sinh_real = sinh_result.real.read()?;

        // Test cosh
        let cosh_result = a.cosh()?;
        let cosh_real = cosh_result.real.read()?;
        assert_relative_eq!(cosh_real[0], 0.0);
        Ok(())
    }
}
