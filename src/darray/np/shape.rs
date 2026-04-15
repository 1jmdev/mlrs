use crate::darray::Array;

/// Reshapes an array.
pub fn reshape(array: &Array, new_shape: &[isize]) -> Array {
    array.reshape(new_shape)
}

/// Transposes an array by reversing its axes.
pub fn transpose(array: &Array) -> Array {
    array.transpose()
}

/// Flattens an array into one dimension.
pub fn flatten(array: &Array) -> Array {
    array.flatten()
}

/// Returns a flattened copy of the array.
pub fn ravel(array: &Array) -> Array {
    array.ravel()
}

/// Permutes axes explicitly.
pub fn permute_axes(array: &Array, axes: &[usize]) -> Array {
    array.permute_axes(axes)
}

/// Swaps two axes.
pub fn swapaxes(array: &Array, axis1: usize, axis2: usize) -> Array {
    array.swapaxes(axis1, axis2)
}

/// Moves one axis to a new position.
pub fn moveaxis(array: &Array, source: usize, destination: usize) -> Array {
    array.moveaxis(source, destination)
}

/// Inserts a length-1 axis at the provided index.
pub fn expand_dims(array: &Array, axis: usize) -> Array {
    array.expand_dims(axis)
}

/// Removes every length-1 axis.
pub fn squeeze(array: &Array) -> Array {
    array.squeeze()
}

/// Concatenates arrays along an axis.
pub fn concatenate(arrays: &[&Array], axis: usize) -> Array {
    Array::concatenate(arrays, axis)
}

/// Stacks arrays along a new axis.
pub fn stack(arrays: &[&Array], axis: usize) -> Array {
    Array::stack(arrays, axis)
}

/// Repeats elements.
pub fn repeat(array: &Array, repeats: usize, axis: Option<usize>) -> Array {
    array.repeat(repeats, axis)
}

/// Tiles an array across axes.
pub fn tile(array: &Array, reps: &[usize]) -> Array {
    array.tile(reps)
}
