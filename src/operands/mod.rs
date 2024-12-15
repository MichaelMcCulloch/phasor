use crate::TensorBase;
use candle_core::WithDType;

mod column_vector;
mod complex_column_vector;
mod complex_matrix;
mod complex_row_vector;
mod complex_scalar;
mod matrix;
mod real_scaler;
mod row_vector;

pub use {
    column_vector::ColumnVector, complex_column_vector::ComplexColumnVector,
    complex_matrix::ComplexMatrix, complex_row_vector::ComplexRowVector,
    complex_scalar::ComplexScalar, matrix::Matrix, real_scaler::Scalar, row_vector::RowVector,
};

#[cfg(test)]
mod test {
    use approx::{assert_relative_eq, RelativeEq};
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
}
