use super::Array;
use super::utils::{PAR_THRESHOLD, axis_inner_outer, index_to_offset};
use rayon::prelude::*;

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
        let output_len = shape.iter().product::<usize>();
        if output_len == 0 {
            return Self::from_shape_vec(&shape, Vec::new());
        }

        if inner == 1 {
            let mut data = vec![0.0; output_len];
            if output_len >= PAR_THRESHOLD {
                data.par_chunks_mut(indices.len())
                    .enumerate()
                    .for_each(|(outer_index, chunk)| {
                        for (position, &index) in indices.iter().enumerate() {
                            assert!(
                                index < axis_len,
                                "index {index} out of bounds for axis {axis}"
                            );
                            chunk[position] = self.data[outer_index * axis_len + index];
                        }
                    });
            } else {
                for (outer_index, chunk) in data.chunks_mut(indices.len()).enumerate() {
                    for (position, &index) in indices.iter().enumerate() {
                        assert!(
                            index < axis_len,
                            "index {index} out of bounds for axis {axis}"
                        );
                        chunk[position] = self.data[outer_index * axis_len + index];
                    }
                }
            }
            return Self::from_shape_vec(&shape, data);
        }

        let mut data = vec![0.0; output_len];

        if output_len >= PAR_THRESHOLD && outer > 1 {
            data.par_chunks_mut(indices.len() * inner)
                .enumerate()
                .for_each(|(outer_index, output_chunk)| {
                    for (position, &index) in indices.iter().enumerate() {
                        assert!(
                            index < axis_len,
                            "index {index} out of bounds for axis {axis}"
                        );
                        let source_start = (outer_index * axis_len + index) * inner;
                        let dest_start = position * inner;
                        output_chunk[dest_start..dest_start + inner]
                            .copy_from_slice(&self.data[source_start..source_start + inner]);
                    }
                });
        } else {
            for outer_index in 0..outer {
                let output_chunk_start = outer_index * indices.len() * inner;
                let output_chunk_end = output_chunk_start + indices.len() * inner;
                let output_chunk = &mut data[output_chunk_start..output_chunk_end];
                for (position, &index) in indices.iter().enumerate() {
                    assert!(
                        index < axis_len,
                        "index {index} out of bounds for axis {axis}"
                    );
                    let source_start = (outer_index * axis_len + index) * inner;
                    let dest_start = position * inner;
                    output_chunk[dest_start..dest_start + inner]
                        .copy_from_slice(&self.data[source_start..source_start + inner]);
                }
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
        let output_len = shape.iter().product::<usize>();
        if output_len == 0 {
            return Self::from_shape_vec(&shape, Vec::new());
        }

        if inner == 1 {
            let mut data = vec![0.0; output_len];
            if output_len >= PAR_THRESHOLD {
                data.par_chunks_mut(slice_len)
                    .enumerate()
                    .for_each(|(outer_index, chunk)| {
                        let start_offset = outer_index * axis_len + start;
                        let end_offset = start_offset + slice_len;
                        chunk.copy_from_slice(&self.data[start_offset..end_offset]);
                    });
            } else {
                for (outer_index, chunk) in data.chunks_mut(slice_len).enumerate() {
                    let start_offset = outer_index * axis_len + start;
                    let end_offset = start_offset + slice_len;
                    chunk.copy_from_slice(&self.data[start_offset..end_offset]);
                }
            }
            return Self::from_shape_vec(&shape, data);
        }

        let mut data = vec![0.0; output_len];
        let output_chunk_len = slice_len * inner;
        if output_len >= PAR_THRESHOLD && outer > 1 {
            data.par_chunks_mut(output_chunk_len)
                .enumerate()
                .for_each(|(outer_index, output_chunk)| {
                    for (position, index) in (start..end).enumerate() {
                        let source_start = (outer_index * axis_len + index) * inner;
                        let dest_start = position * inner;
                        output_chunk[dest_start..dest_start + inner]
                            .copy_from_slice(&self.data[source_start..source_start + inner]);
                    }
                });
        } else {
            for outer_index in 0..outer {
                let output_chunk_start = outer_index * output_chunk_len;
                let output_chunk_end = output_chunk_start + output_chunk_len;
                let output_chunk = &mut data[output_chunk_start..output_chunk_end];
                for (position, index) in (start..end).enumerate() {
                    let source_start = (outer_index * axis_len + index) * inner;
                    let dest_start = position * inner;
                    output_chunk[dest_start..dest_start + inner]
                        .copy_from_slice(&self.data[source_start..source_start + inner]);
                }
            }
        }

        Self::from_shape_vec(&shape, data)
    }
}
