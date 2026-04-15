use super::super::Array;
use super::super::utils::{compute_size, compute_strides, reduced_shape};

impl Array {
    /// Returns the indices of the minimum elements along an axis.
    pub fn argmin_axis(&self, axis: usize) -> Self {
        self.argextreme_axis(axis, |value, best| value < best)
    }

    /// Returns the indices of the maximum elements along an axis.
    pub fn argmax_axis(&self, axis: usize) -> Self {
        self.argextreme_axis(axis, |value, best| value > best)
    }

    fn argextreme_axis<F>(&self, axis: usize, mut better: F) -> Self
    where
        F: FnMut(f64, f64) -> bool,
    {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");

        let shape = reduced_shape(&self.shape, axis);
        let strides = compute_strides(&shape);
        let len = compute_size(&shape);
        let mut best_values = vec![0.0; len];
        let mut best_indices = vec![0.0; len];
        let mut initialized = vec![false; len];

        for (flat_index, &value) in self.data.iter().enumerate() {
            let mut remainder = flat_index;
            let mut reduced_offset = 0;
            let mut reduced_axis = 0;
            let mut axis_coordinate = 0;

            for current_axis in 0..self.ndim() {
                let coordinate = remainder / self.strides[current_axis];
                remainder %= self.strides[current_axis];

                if current_axis == axis {
                    axis_coordinate = coordinate;
                } else {
                    reduced_offset += coordinate * strides[reduced_axis];
                    reduced_axis += 1;
                }
            }

            if !initialized[reduced_offset] || better(value, best_values[reduced_offset]) {
                initialized[reduced_offset] = true;
                best_values[reduced_offset] = value;
                best_indices[reduced_offset] = axis_coordinate as f64;
            }
        }

        Self::from_shape_vec(&shape, best_indices)
    }
}
