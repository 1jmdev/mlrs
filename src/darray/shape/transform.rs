use super::super::Array;
use super::super::utils::{clone_data_parallel, compute_size, compute_strides, infer_shape};
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
            data: clone_data_parallel(&self.data),
            shape: shape.clone(),
            strides: compute_strides(&shape),
        }
    }

    /// Returns a flattened copy of the array.
    pub fn flatten(&self) -> Self {
        Self {
            data: clone_data_parallel(&self.data),
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
        if self.ndim() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut data = vec![0.0; self.data.len()];
            let block = 32usize;

            for row_block in (0..rows).step_by(block) {
                let row_limit = rows.min(row_block + block);
                for col_block in (0..cols).step_by(block) {
                    let col_limit = cols.min(col_block + block);
                    for row in row_block..row_limit {
                        let row_offset = row * cols;
                        for col in col_block..col_limit {
                            data[col * rows + row] = self.data[row_offset + col];
                        }
                    }
                }
            }

            return Self::from_shape_vec(&[cols, rows], data);
        }

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

        if shape.is_empty() {
            data[0] = self.data[0];
            return Self {
                data,
                shape,
                strides,
            };
        }

        let mut mapped_strides = SmallVec::<[usize; 6]>::with_capacity(shape.len());
        for &axis in axes {
            mapped_strides.push(self.strides[axis]);
        }
        let mut coordinates = SmallVec::<[usize; 6]>::from_elem(0, shape.len());
        let mut source_offset = 0usize;

        for value in &mut data {
            *value = self.data[source_offset];

            for axis in (0..shape.len()).rev() {
                coordinates[axis] += 1;
                source_offset += mapped_strides[axis];

                if coordinates[axis] < shape[axis] {
                    break;
                }

                coordinates[axis] = 0;
                source_offset -= shape[axis] * mapped_strides[axis];
            }
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
            data: clone_data_parallel(&self.data),
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
            data: clone_data_parallel(&self.data),
            shape: shape.clone(),
            strides: compute_strides(&shape),
        }
    }
}
