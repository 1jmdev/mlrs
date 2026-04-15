use crate::darray::Array;

/// Returns the total sum of elements or reduces along one axis.
pub fn sum(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.sum_axis(axis),
        None => Array::scalar(array.sum()),
    }
}

/// Returns the total product of elements or reduces along one axis.
pub fn prod(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.prod_axis(axis),
        None => Array::scalar(array.prod()),
    }
}

/// Returns the mean of elements or reduces along one axis.
pub fn mean(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.mean_axis(axis),
        None => Array::scalar(array.mean()),
    }
}

/// Returns the population variance of elements or reduces along one axis.
pub fn var(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.var_axis(axis),
        None => Array::scalar(array.var()),
    }
}

/// Returns the population standard deviation of elements or reduces along one axis.
pub fn std(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.std_axis(axis),
        None => Array::scalar(array.std()),
    }
}

/// Returns the minimum element or reduces along one axis.
pub fn min(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.min_axis(axis),
        None => Array::scalar(array.min()),
    }
}

/// Returns the maximum element or reduces along one axis.
pub fn max(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.max_axis(axis),
        None => Array::scalar(array.max()),
    }
}

/// Returns whether all values are non-zero or reduces along one axis.
pub fn all(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.all_axis(axis),
        None => Array::scalar(if array.all() { 1.0 } else { 0.0 }),
    }
}

/// Returns whether any value is non-zero or reduces along one axis.
pub fn any(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.any_axis(axis),
        None => Array::scalar(if array.any() { 1.0 } else { 0.0 }),
    }
}

/// Returns the flat index of the minimum element or the index along one axis.
pub fn argmin(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.argmin_axis(axis),
        None => Array::scalar(array.argmin() as f64),
    }
}

/// Returns the flat index of the maximum element or the index along one axis.
pub fn argmax(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.argmax_axis(axis),
        None => Array::scalar(array.argmax() as f64),
    }
}

/// Returns the cumulative sum over the flattened array or along one axis.
pub fn cumsum(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.cumsum_axis(axis),
        None => array.cumsum(),
    }
}

/// Returns the cumulative product over the flattened array or along one axis.
pub fn cumprod(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.cumprod_axis(axis),
        None => array.cumprod(),
    }
}
