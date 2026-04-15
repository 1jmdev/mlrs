use crate::darray::Array;

/// Returns the value at a multi-dimensional index.
pub fn get(array: &Array, indices: &[usize]) -> f64 {
    array.get(indices)
}

/// Returns the scalar value stored in a 0-D array or a one-element array.
pub fn item(array: &Array) -> f64 {
    array.item()
}

/// Returns a copied row from a 2-D array.
pub fn row(array: &Array, row: usize) -> Array {
    array.row(row)
}

/// Returns a copied column from a 2-D array.
pub fn column(array: &Array, column: usize) -> Array {
    array.column(column)
}

/// Takes elements along an axis.
pub fn take(array: &Array, indices: &[usize], axis: usize) -> Array {
    array.take(indices, axis)
}

/// Returns a half-open slice along an axis.
pub fn slice_axis(array: &Array, axis: usize, start: usize, end: usize) -> Array {
    array.slice_axis(axis, start, end)
}
