use super::super::Array;
use super::super::utils::{compute_size, compute_strides, index_to_offset, infer_shape};
use smallvec::SmallVec;

impl Array {
    /// Reshapes the array while preserving the flattened data.
    ///
    /// Use `-1` for at most one inferred dimension.
    pub fn reshape(&self, new_shape: &[isize]) -> Self {
        let shape = infer_shape(new_shape, self.data.len());
        assert_eq!(
            compute_size(&shape),
            self.data.len(),
            "cannot reshape {} elements into {:?}",
            self.data.len(),
            new_shape
        );

        Self {
            data: self.data.clone(),
            shape: shape.clone(),
            strides: compute_strides(&shape),
        }
    }

    /// Returns a flattened copy of the array.
    pub fn flatten(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: [self.data.len()].into_iter().collect(),
            strides: compute_strides(&[self.data.len()]),
        }
    }

    /// Returns a flattened copy of the array.
    pub fn ravel(&self) -> Self {
        self.flatten()
    }

    /// Reverses the axes of the array.
    pub fn transpose(&self) -> Self {
        let axes = (0..self.ndim()).rev().collect::<SmallVec<[usize; 6]>>();
        self.permute_axes(&axes)
    }

    /// Permutes the axes using an explicit axis order.
    pub fn permute_axes(&self, axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            self.ndim(),
            "expected {} axes, got {:?}",
            self.ndim(),
            axes
        );

        let mut seen = SmallVec::<[bool; 6]>::from_elem(false, self.ndim());
        for &axis in axes {
            assert!(axis < self.ndim(), "axis {axis} out of bounds");
            assert!(!seen[axis], "duplicate axis {axis} in permutation");
            seen[axis] = true;
        }

        let shape = axes
            .iter()
            .map(|&axis| self.shape[axis])
            .collect::<SmallVec<[usize; 6]>>();
        let strides = compute_strides(&shape);
        let mut data = vec![0.0; self.data.len()];

        for (flat_index, value) in data.iter_mut().enumerate() {
            let mut remainder = flat_index;
            let mut source_indices = SmallVec::<[usize; 6]>::from_elem(0, self.ndim());

            for (target_axis, &stride) in strides.iter().enumerate() {
                let coordinate = if shape.is_empty() {
                    0
                } else {
                    remainder / stride
                };
                if !shape.is_empty() {
                    remainder %= stride;
                }
                source_indices[axes[target_axis]] = coordinate;
            }

            let source_offset = index_to_offset(&source_indices, &self.shape, &self.strides);
            *value = self.data[source_offset];
        }

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Swaps two axes.
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> Self {
        assert!(axis1 < self.ndim(), "axis {axis1} out of bounds");
        assert!(axis2 < self.ndim(), "axis {axis2} out of bounds");

        let mut axes = (0..self.ndim()).collect::<SmallVec<[usize; 6]>>();
        axes.swap(axis1, axis2);
        self.permute_axes(&axes)
    }

    /// Moves one axis to a new position.
    pub fn moveaxis(&self, source: usize, destination: usize) -> Self {
        assert!(source < self.ndim(), "axis {source} out of bounds");
        assert!(
            destination < self.ndim(),
            "axis {destination} out of bounds"
        );

        let mut axes = (0..self.ndim()).collect::<SmallVec<[usize; 6]>>();
        let axis = axes.remove(source);
        axes.insert(destination, axis);
        self.permute_axes(&axes)
    }

    /// Inserts a length-1 axis at the given position.
    pub fn expand_dims(&self, axis: usize) -> Self {
        assert!(axis <= self.ndim(), "axis {axis} out of bounds");
        let mut shape = self.shape.clone();
        shape.insert(axis, 1);

        Self {
            data: self.data.clone(),
            shape: shape.clone(),
            strides: compute_strides(&shape),
        }
    }

    /// Removes all length-1 axes.
    pub fn squeeze(&self) -> Self {
        let shape = self
            .shape
            .iter()
            .copied()
            .filter(|&dimension| dimension != 1)
            .collect::<SmallVec<[usize; 6]>>();

        Self {
            data: self.data.clone(),
            shape: shape.clone(),
            strides: compute_strides(&shape),
        }
    }
}
