use candle_core::{DType, Device, Result, Shape, Tensor, WithDType};
use core::f64;

#[inline]
pub fn generic_add<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.add(rhs)
}

#[inline]
pub fn generic_sub<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.sub(rhs)
}

#[inline]
pub fn generic_mul<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.mul(rhs)
}

#[inline]
pub fn generic_div<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.div(rhs)
}

#[inline]
pub fn generic_clamp<T: WithDType>(tensor: &Tensor, min: &T, max: &T) -> Result<Tensor> {
    tensor.clamp(*min, *max)
}

#[inline]
pub fn generic_add_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_add(&scalar_tensor)
}

#[inline]
pub fn generic_sub_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_sub(&scalar_tensor)
}

#[inline]
pub fn generic_mul_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_mul(&scalar_tensor)
}

#[inline]
pub fn generic_div_scalar<T: WithDType>(tensor: &Tensor, scalar: T) -> Result<Tensor> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), tensor.device())?;
    tensor.broadcast_div(&scalar_tensor)
}

#[inline]
pub fn generic_powf<T: WithDType>(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    tensor.powf(exponent)
}

#[inline]
pub fn generic_pow<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.pow(rhs)
}

#[inline]
pub fn generic_tanh<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.tanh()
}

#[inline]
pub fn generic_neg<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.neg()
}

#[inline]
pub fn generic_abs<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.abs()
}
#[inline]
pub fn generic_exp<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.exp()
}

#[inline]
pub fn generic_log<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.log()
}

#[inline]
pub fn generic_mean<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.mean_all()
}

#[inline]
pub fn generic_lt<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.lt(rhs)
}

#[inline]
pub fn generic_lte<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.le(rhs)
}

#[inline]
pub fn generic_eq<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.eq(rhs)
}

#[inline]
pub fn generic_ne<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(rhs)
}

#[inline]
pub fn generic_gt<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.gt(rhs)
}

#[inline]
pub fn generic_gte<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ge(rhs)
}

#[inline]
pub fn generic_where<T: WithDType>(
    condition: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
) -> Result<Tensor> {
    condition.where_cond(on_true, on_false)
}

#[inline]
pub fn generic_and<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.mul(rhs)
}

#[inline]
pub fn generic_or<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(0u8)?.add(&rhs.ne(0u8)?)?.ne(0u8)
}

#[inline]
pub fn generic_xor<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.ne(rhs)
}

#[inline]
pub fn generic_not<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.eq(0u8)
}

#[inline]
pub fn generic_zeros<T: WithDType>(device: &Device, dtype: DType, shape: &Shape) -> Result<Tensor> {
    Tensor::zeros(shape, dtype, device)
}

#[inline]
pub fn generic_transpose<T: WithDType>(tensor: &Tensor) -> Result<Tensor> {
    tensor.t()
}

#[inline]
pub fn generic_matmul<T: WithDType>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.matmul(rhs)
}
#[inline]
pub fn generic_complex_matmul<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let sub_rhs = generic_matmul::<T>(&lhs_imag, &rhs_imag)?;
    let sub_lhs = generic_matmul::<T>(&lhs_real, &rhs_real)?;
    let add_lhs = generic_matmul::<T>(&lhs_real, &rhs_imag)?;
    let add_rhs = generic_matmul::<T>(&lhs_imag, &rhs_real)?;

    let real = generic_sub::<T>(&sub_lhs, &sub_rhs)?;
    let imag = generic_add::<T>(&add_lhs, &add_rhs)?;

    Ok((real, imag))
}

#[inline]
pub fn generic_sum<T: WithDType>(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    tensor.sum(dim)
}

#[inline]
pub fn generic_broadcast<T: WithDType>(tensor: &Tensor, shape: &Shape) -> Result<Tensor> {
    tensor.broadcast_as(shape)
}

#[inline]
pub fn generic_complex_add<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((lhs_real.add(rhs_real)?, lhs_imag.add(rhs_imag)?))
}

#[inline]
pub fn generic_complex_sub<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((lhs_real.sub(rhs_real)?, lhs_imag.sub(rhs_imag)?))
}

#[inline]
pub fn generic_complex_mul<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let real = lhs_real.mul(rhs_real)?.sub(&lhs_imag.mul(rhs_imag)?)?;
    let imag = lhs_real.mul(rhs_imag)?.add(&lhs_imag.mul(rhs_real)?)?;
    Ok((real, imag))
}

#[inline]
pub fn generic_complex_div<T: WithDType>(
    lhs_real: &Tensor,
    lhs_imag: &Tensor,
    rhs_real: &Tensor,
    rhs_imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let denom = rhs_real.mul(rhs_real)?.add(&rhs_imag.mul(rhs_imag)?)?;
    let real = lhs_real
        .mul(rhs_real)?
        .add(&lhs_imag.mul(rhs_imag)?)?
        .div(&denom)?;
    let imag = lhs_imag
        .mul(rhs_real)?
        .sub(&lhs_real.mul(rhs_imag)?)?
        .div(&denom)?;
    Ok((real, imag))
}

#[inline]
pub fn generic_complex_clamp<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    min: &T,
    max: &T,
) -> Result<(Tensor, Tensor)> {
    Ok((real.clamp(*min, *max)?, imag.clamp(*min, *max)?))
}

#[inline]
pub fn generic_complex_mul_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((
        real.broadcast_mul(&scalar_tensor)?,
        imag.broadcast_mul(&scalar_tensor)?,
    ))
}

#[inline]
pub fn generic_complex_div_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((
        real.broadcast_div(&scalar_tensor)?,
        imag.broadcast_div(&scalar_tensor)?,
    ))
}

#[inline]
pub fn generic_complex_add_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((real.broadcast_add(&scalar_tensor)?, imag.clone()))
}

#[inline]
pub fn generic_complex_sub_scalar<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    scalar: T,
) -> Result<(Tensor, Tensor)> {
    let scalar_tensor = Tensor::from_vec(vec![scalar], Shape::from((1, 1)), real.device())?;
    Ok((real.broadcast_sub(&scalar_tensor)?, imag.clone()))
}

#[inline]
pub fn generic_complex_componentwise_tanh<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((generic_tanh::<T>(&real)?, generic_tanh::<T>(&imag)?))
}

#[inline]
pub fn generic_complex_lt<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_lt::<T>(&mag_self, &mag_other)
}

#[inline]
pub fn generic_complex_lte<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_lte::<T>(&mag_self, &mag_other)
}

#[inline]
pub fn generic_complex_eq<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let real_eq = generic_eq::<T>(real, other_real)?;
    let imag_eq = generic_eq::<T>(imag, other_imag)?;
    generic_mul::<T>(&real_eq, &imag_eq)
}

#[inline]
pub fn generic_complex_ne<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let real_ne = generic_ne::<T>(real, other_real)?;
    let imag_ne = generic_ne::<T>(imag, other_imag)?;
    generic_gt::<T>(
        &generic_add::<T>(&real_ne, &imag_ne)?,
        &generic_zeros::<T>(real.device(), T::DTYPE, real.shape())?,
    )
}

#[inline]
pub fn generic_complex_gt<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_gt::<T>(&mag_self, &mag_other)
}

#[inline]
pub fn generic_complex_gte<T: WithDType>(
    real: &Tensor,
    imag: &Tensor,
    other_real: &Tensor,
    other_imag: &Tensor,
) -> Result<Tensor> {
    let mag_self = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(real, 2.0)?,
            &generic_powf::<T>(imag, 2.0)?,
        )?,
        0.5,
    )?;
    let mag_other = generic_powf::<T>(
        &generic_add::<T>(
            &generic_powf::<T>(other_real, 2.0)?,
            &generic_powf::<T>(other_imag, 2.0)?,
        )?,
        0.5,
    )?;
    generic_gte::<T>(&mag_self, &mag_other)
}

#[cfg(test)]
mod test {
    use crate::utils::methods::*;
    use approx::{assert_relative_eq, RelativeEq};
    use candle_core::{DType, Device, Result, Shape, Tensor};
    use core::f64;
    use std::f64::consts::PI;
    use std::fmt::Debug;
    pub fn assert_relative_eq_vec<T: RelativeEq + Debug>(lhs: Vec<T>, rhs: Vec<T>) {
        lhs.iter()
            .zip(rhs)
            .for_each(|(l, r)| assert_relative_eq!(l, &r));
    }
    pub fn assert_relative_eq_vec_vec<T: RelativeEq + Debug>(lhs: Vec<Vec<T>>, rhs: Vec<Vec<T>>) {
        lhs.iter()
            .flatten()
            .zip(rhs.iter().flatten())
            .for_each(|(l, r)| assert_relative_eq!(l, &r));
    }
    fn create_tensor<T: WithDType>(data: Vec<T>, shape: Shape, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(data, shape, device)
    }

    #[test]
    fn test_generic_add() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let result = generic_add::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_sub() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_sub::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![3.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_mul() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![4.0, 5.0], Shape::from((1, 2)), &device)?;
        let result = generic_mul::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![8.0, 15.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_div() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![10.0, 12.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_div::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![5.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_clamp() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(
            vec![1.0, 5.0, 2.0, 8.0],
            Shape::from(Shape::from((1, 4))),
            &device,
        )?;
        let min = 3.0;
        let max = 6.0;
        let result = generic_clamp::<f64>(&a, &min, &max)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![3.0, 5.0, 3.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_add_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_add_scalar::<f64>(&a, 3.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_sub_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let result = generic_sub_scalar::<f64>(&a, 2.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_mul_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_mul_scalar::<f64>(&a, 4.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![8.0, 12.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_div_scalar() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![10.0, 12.0], Shape::from((1, 2)), &device)?;
        let result = generic_div_scalar::<f64>(&a, 2.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![5.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_powf() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_powf::<f64>(&a, 2.0)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 9.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_pow() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 3.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_pow::<f64>(&a, &b)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![4.0, 9.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_tanh() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_tanh::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 0.7615941559557649);
        Ok(())
    }

    #[test]
    fn test_generic_neg() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, -2.0], Shape::from((1, 2)), &device)?;
        let result = generic_neg::<f64>(&a)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![-1.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_abs() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![-1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_abs::<f64>(&a)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![1.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_exp() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![0.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_exp::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 1.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 2.718281828459045);
        Ok(())
    }

    #[test]
    fn test_generic_log() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_log::<f64>(&a)?;
        assert_relative_eq!(result.to_vec2::<f64>()?[0][0], 0.0);
        assert_relative_eq!(result.to_vec2::<f64>()?[0][1], 0.6931471805599453);
        Ok(())
    }

    #[test]
    fn test_generic_mean() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0], Shape::from((1, 3)), &device)?;
        let result = generic_mean::<f64>(&a)?;
        assert_relative_eq!(result.to_scalar::<f64>()?, 2.0);
        Ok(())
    }

    #[test]
    fn test_generic_lt() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 1.0], Shape::from((1, 2)), &device)?;
        let result = generic_lt::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_lte() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![2.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_lte::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_eq() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_eq::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_ne() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 3.0], Shape::from((1, 2)), &device)?;
        let result = generic_ne::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![0, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_gt() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 1.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_gt::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_gte() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![2.0, 2.0], Shape::from((1, 2)), &device)?;
        let b = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let result = generic_gte::<f64>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_where() -> Result<()> {
        let device = Device::Cpu;
        let condition = create_tensor(vec![1u8, 0u8], Shape::from((1, 2)), &device)?;
        let on_true = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let on_false = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let result = generic_where::<f64>(&condition, &on_true, &on_false)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![1.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_and() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 1u8, 0u8, 0u8], Shape::from((1, 4)), &device)?;
        let b = create_tensor(vec![1u8, 0u8, 1u8, 0u8], Shape::from((1, 4)), &device)?;
        let result = generic_and::<u8>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 0, 0, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_or() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 1u8, 0u8, 0u8], Shape::from((1, 4)), &device)?;
        let b = create_tensor(vec![1u8, 0u8, 1u8, 0u8], Shape::from((1, 4)), &device)?;
        let result = generic_or::<u8>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![1, 1, 1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_xor() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 1u8, 0u8, 0u8], Shape::from((1, 4)), &device)?;
        let b = create_tensor(vec![1u8, 0u8, 1u8, 0u8], Shape::from((1, 4)), &device)?;
        let result = generic_xor::<u8>(&a, &b)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![0, 1, 1, 0]]);
        Ok(())
    }

    #[test]
    fn test_generic_not() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1u8, 0u8], Shape::from((1, 2)), &device)?;
        let result = generic_not::<u8>(&a)?;
        assert_eq!(result.to_vec2::<u8>()?, vec![vec![0, 1]]);
        Ok(())
    }

    #[test]
    fn test_generic_zeros() -> Result<()> {
        let device = Device::Cpu;
        let shape = Shape::from((1, 2));
        let result = generic_zeros::<f64>(&device, DType::F64, &shape)?;
        assert_relative_eq_vec_vec(result.to_vec2::<f64>()?, vec![vec![0.0, 0.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_transpose() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::from((2, 2)), &device)?;
        let result = generic_transpose::<f64>(&a)?;
        assert_eq!(
            result.to_vec2::<f64>()?,
            vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_matmul() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::from((2, 2)), &device)?;
        let b = create_tensor(vec![5.0, 6.0, 7.0, 8.0], Shape::from((2, 2)), &device)?;
        let result = generic_matmul::<f64>(&a, &b)?;
        assert_eq!(
            result.to_vec2::<f64>()?,
            vec![vec![19.0, 22.0], vec![43.0, 50.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_sum() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::from((1, 4)), &device)?;
        let result = generic_sum::<f64>(&a, 1)?;
        assert_eq!(result.to_vec1::<f64>()?, vec![10.0]);
        Ok(())
    }

    #[test]
    fn test_generic_broadcast() -> Result<()> {
        let device = Device::Cpu;
        let a = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let shape = Shape::from(Shape::from((2, 2)));
        let result = generic_broadcast::<f64>(&a, &shape)?;
        assert_eq!(
            result.to_vec2::<f64>()?,
            vec![vec![1.0, 2.0], vec![1.0, 2.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_complex_add() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_add::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq_vec_vec(real.to_vec2::<f64>()?, vec![vec![6.0, 8.0]]);
        assert_relative_eq_vec_vec(imag.to_vec2::<f64>()?, vec![vec![10.0, 12.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_sub() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_sub::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq_vec_vec(real.to_vec2::<f64>()?, vec![vec![4.0, 4.0]]);
        assert_relative_eq_vec_vec(imag.to_vec2::<f64>()?, vec![vec![4.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_mul() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_mul::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq_vec_vec(real.to_vec2::<f64>()?, vec![vec![-16.0, -20.0]]);
        assert_relative_eq_vec_vec(imag.to_vec2::<f64>()?, vec![vec![22.0, 40.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_div() -> Result<()> {
        let device = Device::Cpu;
        let real1 = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag1 = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let real2 = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let imag2 = create_tensor(vec![7.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real, imag) = generic_complex_div::<f64>(&real1, &imag1, &real2, &imag2)?;
        assert_relative_eq!(real.to_vec2::<f64>()?[0][0], 0.35135135135135137);
        assert_relative_eq!(real.to_vec2::<f64>()?[0][1], 0.44);
        assert_relative_eq!(imag.to_vec2::<f64>()?[0][0], 0.1081081081081081);
        assert_relative_eq!(imag.to_vec2::<f64>()?[0][1], 0.08);
        Ok(())
    }

    #[test]
    fn test_generic_complex_clamp() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0, 5.0, 2.0, 8.0], Shape::from((1, 4)), &device)?;
        let imag = create_tensor(vec![1.0, 5.0, 2.0, 8.0], Shape::from((1, 4)), &device)?;
        let min = 3.0;
        let max = 6.0;
        let (real_result, imag_result) = generic_complex_clamp::<f64>(&real, &imag, &min, &max)?;
        assert_eq!(
            real_result.to_vec2::<f64>()?,
            vec![vec![3.0, 5.0, 3.0, 6.0]]
        );
        assert_eq!(
            imag_result.to_vec2::<f64>()?,
            vec![vec![3.0, 5.0, 3.0, 6.0]]
        );
        Ok(())
    }

    #[test]
    fn test_generic_complex_mul_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_mul_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![2.0, 4.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![6.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_div_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![2.0, 4.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![6.0, 8.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_div_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![1.0, 2.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_add_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0, 2.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_add_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_sub_scalar() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![3.0, 4.0], Shape::from((1, 2)), &device)?;
        let imag = create_tensor(vec![5.0, 6.0], Shape::from((1, 2)), &device)?;
        let (real_result, imag_result) = generic_complex_sub_scalar::<f64>(&real, &imag, 2.0)?;
        assert_relative_eq_vec_vec(real_result.to_vec2::<f64>()?, vec![vec![1.0, 2.0]]);
        assert_relative_eq_vec_vec(imag_result.to_vec2::<f64>()?, vec![vec![5.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn test_generic_complex_componentwise_tanh() -> Result<()> {
        let device = Device::Cpu;
        let real = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let imag = create_tensor(vec![1.0], Shape::from((1, 1)), &device)?;
        let (real_result, imag_result) = generic_complex_componentwise_tanh::<f64>(&real, &imag)?;
        assert_relative_eq!(real_result.to_vec2::<f64>()?[0][0], 0.7615941559557649);
        assert_relative_eq!(imag_result.to_vec2::<f64>()?[0][0], 0.7615941559557649);
        Ok(())
    }
}
