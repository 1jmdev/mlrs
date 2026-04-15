use crate::darray::Array;

/// Computes the dot product.
pub fn dot(left: &Array, right: &Array) -> Array {
    left.dot(right)
}

/// Computes matrix multiplication.
pub fn matmul(left: &Array, right: &Array) -> Array {
    left.matmul(right)
}

/// Computes the vector outer product.
pub fn outer(left: &Array, right: &Array) -> Array {
    left.outer(right)
}

/// Computes the flattened dot product.
pub fn vdot(left: &Array, right: &Array) -> Array {
    left.vdot(right)
}

/// Creates or extracts a diagonal.
pub fn diag(array: &Array, diagonal: isize) -> Array {
    array.diag(diagonal)
}

/// Returns a diagonal matrix built from the flattened array.
pub fn diagflat(array: &Array, diagonal: isize) -> Array {
    array.diagflat(diagonal)
}

/// Sums over a matrix diagonal.
pub fn trace(array: &Array, offset: isize) -> Array {
    Array::scalar(array.trace(offset))
}
