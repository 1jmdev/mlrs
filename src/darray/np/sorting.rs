use crate::darray::Array;

pub use crate::darray::SearchSide;

/// Returns a sorted flattened copy of the array.
pub fn sort(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.sort_axis(axis),
        None => array.sort(),
    }
}

/// Returns the indices that would sort the array.
pub fn argsort(array: &Array, axis: Option<usize>) -> Array {
    match axis {
        Some(axis) => array.argsort_axis(axis),
        None => array.argsort(),
    }
}

/// Finds insertion indices in a sorted 1-D array.
pub fn searchsorted(array: &Array, values: &Array, side: SearchSide) -> Array {
    array.searchsorted(values, side)
}

/// Returns per-axis indices of non-zero values.
pub fn nonzero(array: &Array) -> Vec<Array> {
    array.nonzero()
}

/// Returns flat indices of non-zero values.
pub fn flatnonzero(array: &Array) -> Array {
    array.flatnonzero()
}

/// Returns coordinates of non-zero values.
pub fn argwhere(array: &Array) -> Array {
    array.argwhere()
}

/// Returns sorted unique flattened values.
pub fn unique(array: &Array) -> Array {
    array.unique()
}
