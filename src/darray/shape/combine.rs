use super::super::Array;
use super::super::utils::{axis_inner_outer, compute_size, compute_strides};

impl Array {
    /// Concatenates arrays along an existing axis.
    pub fn concatenate(arrays: &[&Self], axis: usize) -> Self {
        assert!(
            !arrays.is_empty(),
            "concatenate() requires at least one array"
        );

        let first = arrays[0];
        assert!(axis < first.ndim(), "axis {axis} out of bounds");

        for array in arrays.iter().skip(1) {
            assert_eq!(
                array.ndim(),
                first.ndim(),
                "concatenate() requires matching dimensionality"
            );
            for current_axis in 0..first.ndim() {
                if current_axis != axis {
                    assert_eq!(
                        array.shape[current_axis], first.shape[current_axis],
                        "concatenate() shape mismatch"
                    );
                }
            }
        }

        let (inner, outer, _) = axis_inner_outer(&first.shape, axis);
        let mut shape = first.shape.clone();
        shape[axis] = arrays.iter().map(|array| array.shape[axis]).sum();

        let mut data = Vec::with_capacity(compute_size(&shape));
        for outer_index in 0..outer {
            for array in arrays {
                let axis_len = array.shape[axis];
                let start = outer_index * axis_len * inner;
                let end = start + axis_len * inner;
                data.extend_from_slice(&array.data[start..end]);
            }
        }

        Self {
            data,
            shape: shape.clone(),
            strides: compute_strides(&shape),
        }
    }

    /// Stacks arrays along a new axis.
    pub fn stack(arrays: &[&Self], axis: usize) -> Self {
        assert!(!arrays.is_empty(), "stack() requires at least one array");
        assert!(axis <= arrays[0].ndim(), "axis {axis} out of bounds");

        for array in arrays.iter().skip(1) {
            assert_eq!(array.shape, arrays[0].shape, "stack() shape mismatch");
        }

        let expanded = arrays
            .iter()
            .map(|array| array.expand_dims(axis))
            .collect::<Vec<_>>();
        let references = expanded.iter().collect::<Vec<_>>();
        Self::concatenate(&references, axis)
    }
}
