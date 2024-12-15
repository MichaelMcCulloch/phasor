mod complex_column_vector;
mod complex_matrix;
mod complex_row_vector;
mod complex_scalar;
mod real_column_vector;
mod real_matrix;
mod real_row_vector;
mod real_scaler;

pub use {
    complex_column_vector::ComplexColumnVector, complex_matrix::ComplexMatrix,
    complex_row_vector::ComplexRowVector, complex_scalar::ComplexScalar,
    real_column_vector::ColumnVector, real_matrix::Matrix, real_row_vector::RowVector,
    real_scaler::Scalar,
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
