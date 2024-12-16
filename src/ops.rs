use candle_core::{DType, Device, FloatDType, Result, WithDType};

// =============== Marker Traits ===============
pub trait IsScalar {}
pub trait IsRowVector<const N: usize> {}
pub trait IsColumnVector<const N: usize> {}
pub trait IsMatrix<const R: usize, const C: usize> {}

// =============== Base Tensor Operations ===============
pub trait TensorBase<T: WithDType>: Sized {
    type ReadOutput;
    fn device(&self) -> &Device;
    fn dtype() -> DType;
    fn shape() -> (usize, usize);
    fn read(&self) -> Result<Self::ReadOutput>;
}

// =============== Element-wise Operations ===============
pub trait ElementWiseOp<T: WithDType>: TensorBase<T> {
    type Output;
    fn add(&self, rhs: &Self) -> Result<Self::Output>;
    fn sub(&self, rhs: &Self) -> Result<Self::Output>;
    fn mul(&self, rhs: &Self) -> Result<Self::Output>;
    fn div(&self, rhs: &Self) -> Result<Self::Output>;
    fn clamp(&self, min: &T, max: &T) -> Result<Self::Output>;
}

// =============== Scalar Operations ===============
pub trait ScalarOp<T: WithDType>: TensorBase<T> {
    fn add_scalar(&self, scalar: T) -> Result<Self>;
    fn sub_scalar(&self, scalar: T) -> Result<Self>;
    fn mul_scalar(&self, scalar: T) -> Result<Self>;
    fn div_scalar(&self, scalar: T) -> Result<Self>;
}
pub trait PowerOp<T: WithDType>: TensorBase<T> {
    fn pow(&self, other: &Self) -> Result<Self>;
}

// =============== Vector Operations ===============
// Row vector operations
pub trait RowVectorOps<T: WithDType, const N: usize>: TensorBase<T> + IsRowVector<N> {
    type DotInput: IsRowVector<N>;
    type DotOutput: IsScalar + TensorBase<T>;
    type MatMulMatrix<const M: usize>: IsMatrix<N, M>;
    type MatMulOutput<const M: usize>: IsRowVector<M>;
    type TransposeOutput: IsColumnVector<N>;
    type BroadcastOutput<const R: usize>: IsMatrix<R, N>;

    fn dot(&self, column_vector: &Self::DotInput) -> Result<Self::DotOutput>;
    fn matmul<const M: usize>(
        &self,
        matrix: &Self::MatMulMatrix<M>,
    ) -> Result<Self::MatMulOutput<M>>;
    fn transpose(&self) -> Result<Self::TransposeOutput>;
    fn broadcast<const R: usize>(&self) -> Result<Self::BroadcastOutput<R>>;
}

// Column vector operations
pub trait ColumnVectorOps<T: WithDType, const C: usize>: TensorBase<T> + IsColumnVector<C> {
    type OuterInput<const R: usize>: IsRowVector<R>;
    type OuterOutput<const R: usize>: IsMatrix<C, R>;
    type TransposeOutput: IsRowVector<C>;
    type BroadcastOutput<const CC: usize>: IsMatrix<C, CC>;

    fn outer<const R: usize>(
        &self,
        row_vector: &Self::OuterInput<R>,
    ) -> Result<Self::OuterOutput<R>>;
    fn transpose(&self) -> Result<Self::TransposeOutput>;
    fn broadcast<const CC: usize>(&self) -> Result<Self::BroadcastOutput<CC>>;
}

// =============== Matrix Operations ===============
pub trait MatrixOps<T: WithDType, const R: usize, const C: usize>:
    IsMatrix<R, C> + TensorBase<T>
{
    type MatMulMatrix<const M: usize>: IsMatrix<C, M>;
    type MatMulOutput<const M: usize>: IsMatrix<R, M>;
    type RowSumOutput: IsColumnVector<R>;
    type ColSumOutput: IsRowVector<C>;
    type TransposeOutput: IsMatrix<C, R>;

    fn matmul<const M: usize>(
        &self,
        other: &Self::MatMulMatrix<M>,
    ) -> Result<Self::MatMulOutput<M>>;
    fn sum_rows(&self) -> Result<Self::RowSumOutput>;
    fn sum_cols(&self) -> Result<Self::ColSumOutput>;
    fn transpose(&self) -> Result<Self::TransposeOutput>;
}

// =============== Unary Operations ===============
pub trait UnaryOp<T: WithDType>: TensorBase<T> {
    type ScalarOutput: IsScalar;

    fn neg(&self) -> Result<Self>;
    fn abs(&self) -> Result<Self>;
    fn exp(&self) -> Result<Self>;
    fn log(&self) -> Result<Self>;
    fn mean(&self) -> Result<Self::ScalarOutput>;
}

// =============== Trigonometric Operations ===============
pub trait TrigOp<T: WithDType>: TensorBase<T> {
    type Output;

    fn tanh(&self) -> Result<Self::Output>;
}

// =============== Complex Number Operations ===============
pub trait ComplexOp<T: WithDType>: TensorBase<T> {
    type Output;
    type RealOutput;

    fn conj(&self) -> Result<Self::Output>;
    fn magnitude(&self) -> Result<Self::RealOutput>;
    fn real(&self) -> Result<Self::RealOutput>;
    fn imaginary(&self) -> Result<Self::RealOutput>;
}

// =============== Real-Complex Operations ===============
pub trait RealComplexOp<T: WithDType>: TensorBase<T> {
    type Output: ComplexOp<T>;
    type Input: TensorBase<T>;
    fn mul_complex(&self, rhs: &Self::Input) -> Result<Self::Output>;
}

// =============== Factory Traits ===============
pub trait TensorFactory<T: WithDType>: Sized {
    fn zeros(device: &Device) -> Result<Self>;
    fn ones(device: &Device) -> Result<Self>;
    fn ones_neg(device: &Device) -> Result<Self>;
}

pub trait MatrixFactory<T: WithDType, const R: usize, const C: usize>:
    TensorFactory<T> + IsMatrix<R, C>
{
    fn eye(device: &Device) -> Result<Self>;
}

pub trait TensorFactoryFloat<F: FloatDType>: TensorFactory<F> {
    fn randn(mean: F, std: F, device: &Device) -> Result<Self>;
    fn randu(low: F, high: F, device: &Device) -> Result<Self>;
}

// =============== Conditional Operations ===============
pub trait ConditionalOp<T: WithDType>: TensorBase<u8> {
    type Output: TensorBase<T>;
    type ComplexOutput: ComplexOp<T>;

    fn promote(&self, dtype: DType) -> Result<Self::Output>;
    fn where_cond(&self, on_true: &Self::Output, on_false: &Self::Output) -> Result<Self::Output>;

    fn where_cond_complex(
        &self,
        on_true: &Self::ComplexOutput,
        on_false: &Self::ComplexOutput,
    ) -> Result<Self::ComplexOutput>;
}

// =============== Boolean Logic Operations ===============
pub trait BooleanOp: TensorBase<u8> {
    type Output: TensorBase<u8>;

    fn and(&self, other: &Self) -> Result<Self::Output>;
    fn or(&self, other: &Self) -> Result<Self::Output>;
    fn xor(&self, other: &Self) -> Result<Self::Output>;
    fn not(&self) -> Result<Self::Output>;
}
// =============== Comparison Operations ===============
pub trait ComparisonOp<T: WithDType>: TensorBase<T> {
    type Output: TensorBase<u8>;

    fn lt(&self, other: &Self) -> Result<Self::Output>;
    fn lte(&self, other: &Self) -> Result<Self::Output>;
    fn eq(&self, other: &Self) -> Result<Self::Output>;
    fn ne(&self, other: &Self) -> Result<Self::Output>;
    fn gt(&self, other: &Self) -> Result<Self::Output>;
    fn gte(&self, other: &Self) -> Result<Self::Output>;
}
