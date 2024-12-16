macro_rules! impl_tensor_base {
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TensorBase<T>
            for $type<T, $rows, $cols>
        {
            type ReadOutput = Vec<Vec<T>>;
            #[inline]
            fn device(&self) -> &Device {
                self.$tensor_field.device()
            }
            #[inline]
            fn dtype() -> DType {
                T::DTYPE
            }
            #[inline]
            fn shape() -> (usize, usize) {
                ($rows, $cols)
            }
            #[inline]
            fn read(&self) -> Result<Self::ReadOutput> {
                self.$tensor_field.to_vec2()
            }
        }
    };
    ($type:ident, $tensor_field:tt, $units:tt) => {
        impl<T: WithDType, const $units: usize> TensorBase<T> for $type<T, $units> {
            type ReadOutput = Vec<T>;
            #[inline]
            fn device(&self) -> &Device {
                self.$tensor_field.device()
            }
            #[inline]
            fn dtype() -> DType {
                T::DTYPE
            }
            #[inline]
            fn shape() -> (usize, usize) {
                (1, $units)
            }
            #[inline]
            fn read(&self) -> Result<Self::ReadOutput> {
                Ok(self
                    .$tensor_field
                    .to_vec2()?
                    .into_iter()
                    .flatten()
                    .collect())
            }
        }
    };
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> TensorBase<T> for $type<T> {
            type ReadOutput = T;
            #[inline]
            fn device(&self) -> &Device {
                self.$tensor_field.device()
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
                Ok(self
                    .$tensor_field
                    .reshape((1, 1))?
                    .to_vec2()?
                    .into_iter()
                    .flatten()
                    .last()
                    .unwrap())
            }
        }
    };
}
macro_rules! impl_elementwise_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> ElementWiseOp<T> for $type<T> {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_add::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_sub::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_mul::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_div::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_clamp::<T>(&self.0, min, max)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ElementWiseOp<T> for $type<T, $rows> {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_add::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_sub::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_mul::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_div::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_clamp::<T>(&self.0, min, max)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ElementWiseOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_add::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_sub::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_mul::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_div::<T>(&self.0, &rhs.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_clamp::<T>(&self.0, min, max)?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_scalar_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> ScalarOp<T> for $type<T> {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_add_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_sub_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_mul_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_div_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ScalarOp<T> for $type<T, $rows> {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_add_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_sub_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_mul_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_div_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ScalarOp<T>
            for $type<T, $rows, $cols>
        {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_add_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_sub_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_mul_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_div_scalar::<T>(&self.0, scalar)?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_power_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> PowerOp<T> for $type<T> {
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_pow::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> PowerOp<T> for $type<T, $rows> {
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_pow::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> PowerOp<T>
            for $type<T, $rows, $cols>
        {
            #[inline]
            fn pow(&self, other: &Self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_pow::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_trig_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> TrigOp<T> for $type<T> {
            type Output = Self;

            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_tanh::<T>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> TrigOp<T> for $type<T, $rows> {
            type Output = Self;
            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_tanh::<T>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TrigOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;
            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_tanh::<T>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_unary_op {
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> UnaryOp<T> for $type<T> {
            type ScalarOutput = Scalar<T>;
            #[inline]
            fn neg(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_neg::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn abs(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_abs::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn exp(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_exp::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn log(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_log::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mean(&self) -> Result<Self::ScalarOutput> {
                Ok(Scalar(
                    crate::utils::methods::generic_mean::<T>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> UnaryOp<T> for $type<T, $rows> {
            type ScalarOutput = Scalar<T>;
            #[inline]
            fn neg(&self) -> Result<Self> {
                Ok($type::<T, $rows>(
                    crate::utils::methods::generic_neg::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn abs(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_abs::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn exp(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_exp::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn log(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_log::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mean(&self) -> Result<Self::ScalarOutput> {
                Ok(Scalar(
                    crate::utils::methods::generic_mean::<T>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> UnaryOp<T>
            for $type<T, $rows, $cols>
        {
            type ScalarOutput = Scalar<T>;
            #[inline]
            fn neg(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_neg::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn abs(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_abs::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn exp(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_exp::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn log(&self) -> Result<Self> {
                Ok(Self(
                    crate::utils::methods::generic_log::<T>(&self.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn mean(&self) -> Result<Self::ScalarOutput> {
                Ok(Scalar(
                    crate::utils::methods::generic_mean::<T>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_comparison_op {
    ($type:ident, $tensor_field:tt, $output_type:ident) => {
        impl<T: WithDType> ComparisonOp<T> for $type<T> {
            type Output = $output_type<u8>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_lt::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_lte::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_eq::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_ne::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_gt::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_gte::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $output_type:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ComparisonOp<T> for $type<T, $rows> {
            type Output = $output_type<u8, $rows>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_lt::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_lte::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_eq::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_ne::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_gt::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_gte::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $output_type:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ComparisonOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = $output_type<u8, $rows, $cols>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_lt::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_lte::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_eq::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_ne::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_gt::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_gte::<T>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_conditional_op {
    ($type:ident, $tensor_field:tt, $output_type:ident, $complex_output_type:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ConditionalOp<T>
            for $type<u8, $rows, $cols>
        {
            type Output = $output_type<T, $rows, $cols>;
            type ComplexOutput = $complex_output_type<T, $rows, $cols>;
            #[inline]
            fn where_cond(
                &self,
                on_true: &Self::Output,
                on_false: &Self::Output,
            ) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_where::<u8>(&self.0, &on_true.0, &on_false.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn where_cond_complex(
                &self,
                on_true: &Self::ComplexOutput,
                on_false: &Self::ComplexOutput,
            ) -> Result<Self::ComplexOutput> {
                Ok($complex_output_type {
                    real: self.where_cond(&on_true.real()?, &on_false.real()?)?,
                    imag: self.where_cond(&on_true.imaginary()?, &on_false.imaginary()?)?,
                })
            }
            #[inline]
            fn promote(&self, dtype: DType) -> Result<Self::Output> {
                Ok($output_type(self.0.to_dtype(dtype)?, PhantomData))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $output_type:ident, $complex_output_type:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ConditionalOp<T> for $type<u8, $rows> {
            type Output = $output_type<T, $rows>;
            type ComplexOutput = $complex_output_type<T, $rows>;
            #[inline]
            fn where_cond(
                &self,
                on_true: &Self::Output,
                on_false: &Self::Output,
            ) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_where::<u8>(&self.0, &on_true.0, &on_false.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn where_cond_complex(
                &self,
                on_true: &Self::ComplexOutput,
                on_false: &Self::ComplexOutput,
            ) -> Result<Self::ComplexOutput> {
                Ok($complex_output_type {
                    real: self.where_cond(&on_true.real()?, &on_false.real()?)?,
                    imag: self.where_cond(&on_true.imaginary()?, &on_false.imaginary()?)?,
                })
            }
            #[inline]
            fn promote(&self, dtype: DType) -> Result<Self::Output> {
                Ok($output_type(self.0.to_dtype(dtype)?, PhantomData))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $output_type:ident, $complex_output_type:ident) => {
        impl<T: WithDType> ConditionalOp<T> for $type<u8> {
            type Output = $output_type<T>;
            type ComplexOutput = $complex_output_type<T>;
            #[inline]
            fn where_cond(
                &self,
                on_true: &Self::Output,
                on_false: &Self::Output,
            ) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_where::<u8>(&self.0, &on_true.0, &on_false.0)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn where_cond_complex(
                &self,
                on_true: &Self::ComplexOutput,
                on_false: &Self::ComplexOutput,
            ) -> Result<Self::ComplexOutput> {
                Ok($complex_output_type {
                    real: self.where_cond(&on_true.real()?, &on_false.real()?)?,
                    imag: self.where_cond(&on_true.imaginary()?, &on_false.imaginary()?)?,
                })
            }
            #[inline]
            fn promote(&self, dtype: DType) -> Result<Self::Output> {
                Ok($output_type(self.0.to_dtype(dtype)?, PhantomData))
            }
        }
    };
}
macro_rules! impl_boolean_op {
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<const $rows: usize, const $cols: usize> BooleanOp for $type<u8, $rows, $cols> {
            type Output = Self;
            #[inline]
            fn and(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_and::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn or(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_or::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn xor(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_xor::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn not(&self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_not::<u8>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<const $rows: usize> BooleanOp for $type<u8, $rows> {
            type Output = Self;
            #[inline]
            fn and(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_and::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn or(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_or::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn xor(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_xor::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn not(&self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_not::<u8>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt) => {
        impl BooleanOp for $type<u8> {
            type Output = Self;
            #[inline]
            fn and(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_and::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn or(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_or::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn xor(&self, other: &Self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_xor::<u8>(&self.0, &other.0)?,
                    PhantomData,
                ))
            }

            #[inline]
            fn not(&self) -> Result<Self::Output> {
                Ok(Self(
                    crate::utils::methods::generic_not::<u8>(&self.0)?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_tensor_factory {
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TensorFactory<T>
            for $type<T, $rows, $cols>
        {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::zeros(Self::shape(), T::DTYPE, device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::ones(Self::shape(), T::DTYPE, device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::ones(Self::shape(), T::DTYPE, device)?.neg()?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> TensorFactory<T> for $type<T, $rows> {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::zeros(Self::shape(), T::DTYPE, device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::ones(Self::shape(), T::DTYPE, device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::ones(Self::shape(), T::DTYPE, device)?.neg()?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt) => {
        impl<T: WithDType> TensorFactory<T> for $type<T> {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::zeros(Self::shape(), T::DTYPE, device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::ones(Self::shape(), T::DTYPE, device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::ones(Self::shape(), T::DTYPE, device)?.neg()?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TensorFactory<T>
            for $type<T, $rows, $cols>
        {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::zeros(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones_neg(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> TensorFactory<T> for $type<T, $rows> {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::zeros(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones_neg(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident) => {
        impl<T: WithDType> TensorFactory<T> for $type<T> {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::zeros(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones_neg(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
        }
    };
}
macro_rules! impl_tensor_factory_float {
    ($type:ident, $tensor_field:tt, $rows:tt, $cols:tt) => {
        impl<F: FloatDType, const $rows: usize, const $cols: usize> TensorFactoryFloat<F>
            for $type<F, $rows, $cols>
        {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::randn(mean, std, Self::shape(), device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::rand(low, high, Self::shape(), device)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt, $rows:tt) => {
        impl<F: FloatDType, const $rows: usize> TensorFactoryFloat<F> for $type<F, $rows> {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::randn(mean, std, Self::shape(), device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::rand(low, high, Self::shape(), device)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $tensor_field:tt) => {
        impl<F: FloatDType> TensorFactoryFloat<F> for $type<F> {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::randn(mean, std, Self::shape(), device)?,
                    PhantomData,
                ))
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self(
                    Tensor::rand(low, high, Self::shape(), device)?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<F: FloatDType, const $rows: usize, const $cols: usize> TensorFactoryFloat<F>
            for $type<F, $rows, $cols>
        {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randn(mean, std, device)?,
                    imag: $real_output::randn(mean, std, device)?,
                })
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randu(low, high, device)?,
                    imag: $real_output::randu(low, high, device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident, $rows:tt) => {
        impl<F: FloatDType, const $rows: usize> TensorFactoryFloat<F> for $type<F, $rows> {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randn(mean, std, device)?,
                    imag: $real_output::randn(mean, std, device)?,
                })
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randu(low, high, device)?,
                    imag: $real_output::randu(low, high, device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:tt, $imag_field:tt, $real_output:ident) => {
        impl<F: FloatDType> TensorFactoryFloat<F> for $type<F> {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randn(mean, std, device)?,
                    imag: $real_output::randn(mean, std, device)?,
                })
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randu(low, high, device)?,
                    imag: $real_output::randu(low, high, device)?,
                })
            }
        }
    };
}

macro_rules! impl_tensor_base_complex {
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TensorBase<T>
            for $type<T, $rows, $cols>
        {
            type ReadOutput = (Vec<Vec<T>>, Vec<Vec<T>>);
            #[inline]
            fn device(&self) -> &Device {
                self.$real_field.device()
            }
            #[inline]
            fn dtype() -> DType {
                T::DTYPE
            }
            #[inline]
            fn shape() -> (usize, usize) {
                ($rows, $cols)
            }
            #[inline]
            fn read(&self) -> Result<Self::ReadOutput> {
                Ok((self.$real_field.read()?, self.$imag_field.read()?))
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> TensorBase<T> for $type<T, $rows> {
            type ReadOutput = (Vec<T>, Vec<T>);
            #[inline]
            fn device(&self) -> &Device {
                self.$real_field.device()
            }
            #[inline]
            fn dtype() -> DType {
                T::DTYPE
            }
            #[inline]
            fn shape() -> (usize, usize) {
                (1, $rows)
            }
            #[inline]
            fn read(&self) -> Result<Self::ReadOutput> {
                Ok((self.$real_field.read()?, self.$imag_field.read()?))
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident) => {
        impl<T: WithDType> TensorBase<T> for $type<T> {
            type ReadOutput = (T, T);
            #[inline]
            fn device(&self) -> &Device {
                self.$real_field.device()
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
                Ok((self.$real_field.read()?, self.$imag_field.read()?))
            }
        }
    };
}
macro_rules! impl_complex_op {
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident) => {
        impl<T: WithDType> ComplexOp<T> for $type<T> {
            type Output = Self;
            type RealOutput = $real_output<T>;
            #[inline]
            fn conj(&self) -> Result<Self::Output> {
                Ok(Self {
                    real: self.$real_field.clone(),
                    imag: $real_output::<T>(
                        crate::utils::methods::generic_neg::<T>(&self.$imag_field.0)?,
                        PhantomData,
                    ),
                })
            }
            #[inline]
            fn magnitude(&self) -> Result<Self::RealOutput> {
                Ok($real_output(
                    crate::utils::methods::generic_powf::<T>(
                        &crate::utils::methods::generic_add::<T>(
                            &crate::utils::methods::generic_powf::<T>(&self.$real_field.0, 2.0)?,
                            &crate::utils::methods::generic_powf::<T>(&self.$imag_field.0, 2.0)?,
                        )?,
                        0.5,
                    )?,
                    PhantomData,
                ))
            }

            #[inline]
            fn real(&self) -> Result<Self::RealOutput> {
                Ok(self.$real_field.clone())
            }
            #[inline]
            fn imaginary(&self) -> Result<Self::RealOutput> {
                Ok(self.$imag_field.clone())
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ComplexOp<T> for $type<T, $rows> {
            type Output = Self;
            type RealOutput = $real_output<T, $rows>;
            #[inline]
            fn conj(&self) -> Result<Self::Output> {
                Ok(Self {
                    real: self.$real_field.clone(),
                    imag: $real_output::<T, $rows>(
                        crate::utils::methods::generic_neg::<T>(&self.$imag_field.0)?,
                        PhantomData,
                    ),
                })
            }
            #[inline]
            fn magnitude(&self) -> Result<Self::RealOutput> {
                Ok($real_output::<T, $rows>(
                    crate::utils::methods::generic_powf::<T>(
                        &crate::utils::methods::generic_add::<T>(
                            &crate::utils::methods::generic_powf::<T>(&self.$real_field.0, 2.0)?,
                            &crate::utils::methods::generic_powf::<T>(&self.$imag_field.0, 2.0)?,
                        )?,
                        0.5,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn real(&self) -> Result<Self::RealOutput> {
                Ok(self.$real_field.clone())
            }
            #[inline]
            fn imaginary(&self) -> Result<Self::RealOutput> {
                Ok(self.$imag_field.clone())
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ComplexOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;
            type RealOutput = $real_output<T, $rows, $cols>;
            #[inline]
            fn conj(&self) -> Result<Self::Output> {
                Ok(Self {
                    real: self.$real_field.clone(),
                    imag: $real_output::<T, $rows, $cols>(
                        crate::utils::methods::generic_neg::<T>(&self.$imag_field.0)?,
                        PhantomData,
                    ),
                })
            }
            #[inline]
            fn magnitude(&self) -> Result<Self::RealOutput> {
                Ok($real_output::<T, $rows, $cols>(
                    crate::utils::methods::generic_powf::<T>(
                        &crate::utils::methods::generic_add::<T>(
                            &crate::utils::methods::generic_powf::<T>(&self.$real_field.0, 2.0)?,
                            &crate::utils::methods::generic_powf::<T>(&self.$imag_field.0, 2.0)?,
                        )?,
                        0.5,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn real(&self) -> Result<Self::RealOutput> {
                Ok(self.$real_field.clone())
            }
            #[inline]
            fn imaginary(&self) -> Result<Self::RealOutput> {
                Ok(self.$imag_field.clone())
            }
        }
    };
}
macro_rules! impl_complex_elementwise_op {
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident) => {
        impl<T: WithDType> ElementWiseOp<T> for $type<T> {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_add::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_sub::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_mul::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_div::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_clamp::<T>(
                    &self.real.0,
                    &self.imag.0,
                    min,
                    max,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ElementWiseOp<T> for $type<T, $rows> {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_add::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_sub::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_mul::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_div::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_clamp::<T>(
                    &self.real.0,
                    &self.imag.0,
                    min,
                    max,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ElementWiseOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;
            #[inline]
            fn add(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_add::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_sub::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_mul::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_div::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    &rhs.real.0,
                    &rhs.imag.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn clamp(&self, min: &T, max: &T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_clamp::<T>(
                    &self.real.0,
                    &self.imag.0,
                    min,
                    max,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
}
macro_rules! impl_complex_scalar_op {
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident) => {
        impl<T: WithDType> ScalarOp<T> for $type<T> {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_add_scalar::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_sub_scalar::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_mul_scalar::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_div_scalar::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ScalarOp<T> for $type<T, $rows> {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_add_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_sub_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_mul_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_div_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ScalarOp<T>
            for $type<T, $rows, $cols>
        {
            #[inline]
            fn add_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_add_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn sub_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_sub_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn mul_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_mul_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
            #[inline]
            fn div_scalar(&self, scalar: T) -> Result<Self> {
                let (real, imag) = crate::utils::methods::generic_complex_div_scalar::<T>(
                    &self.real.0,
                    &self.imag.0,
                    scalar,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
}
macro_rules! impl_complex_trig_op {
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident) => {
        impl<T: WithDType> TrigOp<T> for $type<T> {
            type Output = Self;

            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_componentwise_tanh::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> TrigOp<T> for $type<T, $rows> {
            type Output = Self;

            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_componentwise_tanh::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TrigOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = Self;

            #[inline]
            fn tanh(&self) -> Result<Self::Output> {
                let (real, imag) = crate::utils::methods::generic_complex_componentwise_tanh::<T>(
                    &self.$real_field.0,
                    &self.$imag_field.0,
                )?;
                Ok(Self {
                    real: $real_output(real, PhantomData),
                    imag: $real_output(imag, PhantomData),
                })
            }
        }
    };
}
macro_rules! impl_complex_comparison_op {
    ($type:ident, $real_field:ident, $imag_field:ident, $output_type:ident) => {
        impl<T: WithDType> ComparisonOp<T> for $type<T> {
            type Output = $output_type<u8>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok(Scalar(
                    crate::utils::methods::generic_complex_lt::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok(Scalar(
                    crate::utils::methods::generic_complex_lte::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok(Scalar(
                    crate::utils::methods::generic_complex_eq::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok(Scalar(
                    crate::utils::methods::generic_complex_ne::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok(Scalar(
                    crate::utils::methods::generic_complex_gt::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok(Scalar(
                    crate::utils::methods::generic_complex_gte::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $output_type:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> ComparisonOp<T> for $type<T, $rows> {
            type Output = $output_type<u8, $rows>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_complex_lt::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_complex_lte::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_complex_eq::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_complex_ne::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_complex_gt::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok($output_type(
                    crate::utils::methods::generic_complex_gte::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $output_type:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> ComparisonOp<T>
            for $type<T, $rows, $cols>
        {
            type Output = $output_type<u8, $rows, $cols>;
            #[inline]
            fn lt(&self, other: &Self) -> Result<Self::Output> {
                Ok(Matrix(
                    crate::utils::methods::generic_complex_lt::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn lte(&self, other: &Self) -> Result<Self::Output> {
                Ok(Matrix(
                    crate::utils::methods::generic_complex_lte::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn eq(&self, other: &Self) -> Result<Self::Output> {
                Ok(Matrix(
                    crate::utils::methods::generic_complex_eq::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn ne(&self, other: &Self) -> Result<Self::Output> {
                Ok(Matrix(
                    crate::utils::methods::generic_complex_ne::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gt(&self, other: &Self) -> Result<Self::Output> {
                Ok(Matrix(
                    crate::utils::methods::generic_complex_gt::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
            #[inline]
            fn gte(&self, other: &Self) -> Result<Self::Output> {
                Ok(Matrix(
                    crate::utils::methods::generic_complex_gte::<T>(
                        &self.real.0,
                        &self.imag.0,
                        &other.real.0,
                        &other.imag.0,
                    )?,
                    PhantomData,
                ))
            }
        }
    };
}
macro_rules! impl_complex_tensor_factory {
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<T: WithDType, const $rows: usize, const $cols: usize> TensorFactory<T>
            for $type<T, $rows, $cols>
        {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::zeros(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones_neg(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt) => {
        impl<T: WithDType, const $rows: usize> TensorFactory<T> for $type<T, $rows> {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::zeros(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones_neg(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident) => {
        impl<T: WithDType> TensorFactory<T> for $type<T> {
            #[inline]
            fn zeros(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::zeros(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
            #[inline]
            fn ones_neg(device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::ones_neg(device)?,
                    imag: $real_output::zeros(device)?,
                })
            }
        }
    };
}
macro_rules! impl_complex_tensor_factory_float {
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt, $cols:tt) => {
        impl<F: FloatDType, const $rows: usize, const $cols: usize> TensorFactoryFloat<F>
            for $type<F, $rows, $cols>
        {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randn(mean, std, device)?,
                    imag: $real_output::randn(mean, std, device)?,
                })
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randu(low, high, device)?,
                    imag: $real_output::randu(low, high, device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident, $rows:tt) => {
        impl<F: FloatDType, const $rows: usize> TensorFactoryFloat<F> for $type<F, $rows> {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randn(mean, std, device)?,
                    imag: $real_output::randn(mean, std, device)?,
                })
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randu(low, high, device)?,
                    imag: $real_output::randu(low, high, device)?,
                })
            }
        }
    };
    ($type:ident, $real_field:ident, $imag_field:ident, $real_output:ident) => {
        impl<F: FloatDType> TensorFactoryFloat<F> for $type<F> {
            #[inline]
            fn randn(mean: F, std: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randn(mean, std, device)?,
                    imag: $real_output::randn(mean, std, device)?,
                })
            }
            #[inline]
            fn randu(low: F, high: F, device: &Device) -> Result<Self> {
                Ok(Self {
                    real: $real_output::randu(low, high, device)?,
                    imag: $real_output::randu(low, high, device)?,
                })
            }
        }
    };
}
