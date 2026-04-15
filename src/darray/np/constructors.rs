use crate::darray::Array;

/// Creates a one-dimensional array from a slice.
pub fn array(data: &[f64]) -> Array {
    Array::array(data)
}

/// Creates a scalar array.
pub fn scalar(value: f64) -> Array {
    Array::scalar(value)
}

/// Creates an array from flattened data and a shape.
pub fn from_shape_vec(shape: &[usize], data: Vec<f64>) -> Array {
    Array::from_shape_vec(shape, data)
}

/// Creates a zero-filled array.
pub fn zeros(shape: &[usize]) -> Array {
    Array::zeros(shape)
}

/// Creates a one-filled array.
pub fn ones(shape: &[usize]) -> Array {
    Array::ones(shape)
}

/// Creates a filled array.
pub fn full(shape: &[usize], fill_value: f64) -> Array {
    Array::full(shape, fill_value)
}

/// Creates a NaN-filled array for later assignment.
pub fn empty(shape: &[usize]) -> Array {
    Array::empty(shape)
}

/// Creates a range array with `start`, `stop`, and `step`.
pub fn arange(start: f64, stop: f64, step: f64) -> Array {
    Array::arange(start, stop, step)
}

/// Creates evenly spaced values across an interval.
pub fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Array {
    Array::linspace(start, stop, num, endpoint)
}

/// Creates a 2-D array with a unit diagonal.
pub fn eye(rows: usize, cols: usize, diagonal: isize) -> Array {
    Array::eye(rows, cols, diagonal)
}

/// Creates a square identity matrix.
pub fn identity(size: usize) -> Array {
    Array::identity(size)
}

/// Creates a zero-filled array with the same shape as `other`.
pub fn zeros_like(other: &Array) -> Array {
    Array::zeros_like(other)
}

/// Creates a one-filled array with the same shape as `other`.
pub fn ones_like(other: &Array) -> Array {
    Array::ones_like(other)
}

/// Creates a filled array with the same shape as `other`.
pub fn full_like(other: &Array, fill_value: f64) -> Array {
    Array::full_like(other, fill_value)
}

/// Creates a NaN-filled array with the same shape as `other`.
pub fn empty_like(other: &Array) -> Array {
    Array::empty_like(other)
}

/// Returns a deep copy of the array.
pub fn copy(array: &Array) -> Array {
    array.copy()
}
