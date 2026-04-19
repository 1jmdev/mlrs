use crate::darray::Array;
use crate::darray::DArrayError;

/// Computes the dot product.
pub fn dot(left: &Array, right: &Array) -> Result<Array, DArrayError> {
    left.dot(right)
}

/// Computes matrix multiplication.
pub fn matmul(left: &Array, right: &Array) -> Result<Array, DArrayError> {
    left.matmul(right)
}

/// Computes the vector outer product.
pub fn outer(left: &Array, right: &Array) -> Result<Array, DArrayError> {
    left.outer(right)
}

/// Computes the flattened dot product.
pub fn vdot(left: &Array, right: &Array) -> Result<Array, DArrayError> {
    left.vdot(right)
}

/// Creates or extracts a diagonal.
pub fn diag(array: &Array, diagonal: isize) -> Result<Array, DArrayError> {
    array.diag(diagonal)
}

/// Returns a diagonal matrix built from the flattened array.
pub fn diagflat(array: &Array, diagonal: isize) -> Result<Array, DArrayError> {
    array.diagflat(diagonal)
}

/// Sums over a matrix diagonal.
pub fn trace(array: &Array, offset: isize) -> Result<Array, DArrayError> {
    Ok(Array::scalar(array.trace(offset)?))
}
