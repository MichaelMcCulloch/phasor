use crate::ops::*;
use crate::Scalar;
use candle_core::{DType, Device, FloatDType, Result, WithDType};
use std::{f64::consts::PI, marker::PhantomData};
#[derive(Debug, Clone)]
pub struct ComplexScalar<T: WithDType> {
    pub(crate) real: Scalar<T>,
    pub(crate) imag: Scalar<T>,
}

// In src/operands/complex_scalar.rs
impl_complex_op!(ComplexScalar, real, imag, Scalar);

impl<T: WithDType> IsScalar for ComplexScalar<T> {}
impl<T: WithDType> TensorBase<T> for ComplexScalar<T> {
    type ReadOutput = (T, T);
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
        (1, 1)
    }
    #[inline]
    fn read(&self) -> Result<Self::ReadOutput> {
        Ok((self.real.read()?, self.imag.read()?))
    }
}
impl<T: WithDType> ComplexScalar<T> {
    pub fn new(real: T, imag: T, device: &Device) -> Result<ComplexScalar<T>> {
        Ok(Self {
            real: Scalar::new(real, device)?,
            imag: Scalar::new(imag, device)?,
        })
    }
}
impl<T: WithDType> ElementWiseOp<T> for ComplexScalar<T> {
    type Output = ComplexScalar<T>;
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
impl<T: WithDType> ScalarOp<T> for ComplexScalar<T> {
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
        let r = self.magnitude()?;
        let theta = self.arg()?;
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
    #[inline]
    fn pow_scalar(&self, scalar: T) -> Result<Self> {
        let r = self.magnitude()?;
        let theta = self.arg()?;
        let r_pow = r.pow_scalar(scalar)?;
        let new_theta = theta.mul_scalar(scalar)?;
        Ok(Self {
            real: r_pow.mul(&new_theta.cos()?)?,
            imag: r_pow.mul(&new_theta.sin()?)?,
        })
    }
}
impl<T: WithDType> TrigOp<T> for ComplexScalar<T> {
    type Output = ComplexScalar<T>;
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
        Ok(Self {
            real: self.real.tanh()?,
            imag: self.imag.tanh()?,
        })
    }
    #[inline]
    fn atan(&self) -> Result<Self::Output> {
        let zero = Scalar::zeros(self.device())?;
        let one = Scalar::ones(self.device())?;
        let eps = Scalar::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
        let pi_half = Scalar::ones(self.device())?.mul_scalar(T::from_f64(PI / 2.0))?;

        // Form iz
        let iz = ComplexScalar {
            real: self.imag.neg()?,
            imag: self.real.clone(),
        };

        // Compute (1 + iz)/(1 - iz)
        let numerator = ComplexScalar {
            real: one.clone(),
            imag: zero.clone(),
        }
        .add(&iz)?;

        let denominator = ComplexScalar {
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
            ComplexScalar {
                real: log_result.imag.mul_scalar(T::from_f64(0.5))?,
                imag: log_result.real.mul_scalar(T::from_f64(-0.5))?,
            }
        };

        // Special case results
        let zero_result = ComplexScalar {
            real: zero.clone(),
            imag: zero.clone(),
        };

        let i_result = ComplexScalar {
            real: pi_half.mul(&self.imag.div(&mag_imag)?)?,
            imag: Scalar::ones(self.device())?.mul_scalar(T::from_f64(f64::INFINITY))?,
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
        let zero = Scalar::zeros(self.device())?;
        let eps = Scalar::ones(self.device())?.mul_scalar(T::from_f64(1e-15))?;
        let pi = Scalar::ones(self.device())?.mul_scalar(T::from_f64(PI))?;
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
        let adjusted_result = ComplexScalar {
            real: base_atan.real.add(&adjustment)?,
            imag: base_atan.imag,
        };

        // Handle x = 0 cases
        let x_zero_result = ComplexScalar {
            real: y_gte_zero.where_cond(&pi_half, &pi_half.neg()?)?,
            imag: zero.clone(),
        };

        // Combine all cases
        let result = both_zero.where_cond_complex(
            &ComplexScalar {
                real: zero.clone(),
                imag: zero.clone(),
            },
            &x_zero.where_cond_complex(&x_zero_result, &adjusted_result)?,
        )?;

        Ok(result)
    }
}
impl<T: WithDType> UnaryOp<T> for ComplexScalar<T> {
    type TransposeOutput = ComplexScalar<T>;
    type ScalarOutput = ComplexScalar<T>;
    #[inline]
    fn neg(&self) -> Result<Self> {
        Ok(ComplexScalar {
            real: self.real.neg()?,
            imag: self.imag.neg()?,
        })
    }
    #[inline]
    fn abs(&self) -> Result<Self> {
        Ok(Self {
            real: self.real.abs()?,
            imag: Scalar::zeros(self.real.device())?,
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
        Ok(Self {
            real: self.magnitude()?.log()?,
            imag: self.arg()?,
        })
    }
    #[inline]
    fn mean(&self) -> Result<Self::ScalarOutput> {
        Ok(self.clone())
    }
}
impl<T: WithDType> ComparisonOp<T> for ComplexScalar<T> {
    type Output = Scalar<u8>;
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
            .gt(&Scalar::<u8>::zeros(self.device())?)
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
impl<T: WithDType> TensorFactory<T> for ComplexScalar<T> {
    #[inline]
    fn zeros(device: &Device) -> Result<Self> {
        Ok(Self {
            real: Scalar::zeros(device)?,
            imag: Scalar::zeros(device)?,
        })
    }
    #[inline]
    fn ones(device: &Device) -> Result<Self> {
        Ok(Self {
            real: Scalar::ones(device)?,
            imag: Scalar::zeros(device)?,
        })
    }
    #[inline]
    fn ones_neg(device: &Device) -> Result<Self> {
        Ok(Self {
            real: Scalar::ones_neg(device)?,
            imag: Scalar::zeros(device)?,
        })
    }
}
impl<F: FloatDType> TensorFactoryFloat<F> for ComplexScalar<F> {
    #[inline]
    fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
        Ok(Self {
            real: Scalar::<F>::randn(mean, std, device)?,
            imag: Scalar::<F>::randn(mean, std, device)?,
        })
    }
    #[inline]
    fn randu(low: F, high: F, device: &Device) -> Result<Self> {
        Ok(Self {
            real: Scalar::<F>::randu(low, high, device)?,
            imag: Scalar::<F>::randu(low, high, device)?,
        })
    }
}
impl<T: WithDType> RealComplexOp<T> for ComplexScalar<T> {
    type Output = ComplexScalar<T>;
    type Input = Scalar<T>;
    #[inline]
    fn mul_complex(&self, rhs: &Self::Input) -> Result<Self::Output> {
        Ok(ComplexScalar {
            real: self.real.mul(rhs)?,
            imag: self.imag.mul(rhs)?,
        })
    }
}
