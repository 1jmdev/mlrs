use super::Array;
use super::utils::{axis_inner_outer, index_to_offset};

impl Array {
    /// Returns the value at a multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> f64 {
        let offset = index_to_offset(indices, &self.shape, &self.strides);
        self.data[offset]
    }

    /// Writes a value at a multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: f64) {
        let offset = index_to_offset(indices, &self.shape, &self.strides);
        self.data[offset] = value;
    }

    /// Returns the scalar value stored in a 0-D array or a one-element array.
    pub fn item(&self) -> f64 {
        assert!(
            self.data.len() == 1,
            "item() requires exactly one element, got {}",
            self.data.len()
        );
        self.data[0]
    }

    /// Returns a copied row from a 2-D array.
    pub fn row(&self, row: usize) -> Self {
        assert_eq!(self.ndim(), 2, "row() requires a 2-D array");
        let cols = self.shape[1];
        assert!(row < self.shape[0], "row {row} out of bounds");

        let start = row * cols;
        let end = start + cols;
        Self::array(&self.data[start..end])
    }

    /// Returns a copied column from a 2-D array.
    pub fn column(&self, column: usize) -> Self {
        assert_eq!(self.ndim(), 2, "column() requires a 2-D array");
        let rows = self.shape[0];
        let cols = self.shape[1];
        assert!(column < cols, "column {column} out of bounds");

        let mut data = Vec::with_capacity(rows);
        for row in 0..rows {
            data.push(self.data[row * cols + column]);
        }

        Self::array(&data)
    }

    /// Takes elements along one axis using explicit indices.
    pub fn take(&self, indices: &[usize], axis: usize) -> Self {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");
        let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);

        let mut shape = self.shape.clone();
        shape[axis] = indices.len();
        let mut data = Vec::with_capacity(shape.iter().product());

        for outer_index in 0..outer {
            for &index in indices {
                assert!(
                    index < axis_len,
                    "index {index} out of bounds for axis {axis}"
                );
                let start = (outer_index * axis_len + index) * inner;
                let end = start + inner;
                data.extend_from_slice(&self.data[start..end]);
            }
        }

        Self::from_shape_vec(&shape, data)
    }

    /// Returns a half-open slice along one axis.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> Self {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");
        assert!(start <= end, "slice start must be <= end");
        assert!(end <= self.shape[axis], "slice end out of bounds");

        let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);
        let slice_len = end - start;
        let mut shape = self.shape.clone();
        shape[axis] = slice_len;

        if inner == 1 {
            let mut data = Vec::with_capacity(shape.iter().product());
            for outer_index in 0..outer {
                let start_offset = (outer_index * axis_len + start) * inner;
                let end_offset = start_offset + slice_len;
                data.extend_from_slice(&self.data[start_offset..end_offset]);
            }
            return Self::from_shape_vec(&shape, data);
        }

        let mut data = Vec::with_capacity(shape.iter().product());
        for outer_index in 0..outer {
            for index in start..end {
                let start_offset = (outer_index * axis_len + index) * inner;
                let end_offset = start_offset + inner;
                data.extend_from_slice(&self.data[start_offset..end_offset]);
            }
        }

        Self::from_shape_vec(&shape, data)
    }
}
