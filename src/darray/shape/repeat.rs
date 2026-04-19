use super::super::Array;
use super::super::utils::{axis_inner_outer, clone_data_parallel, compute_size, compute_strides};
use smallvec::SmallVec;

impl Array {
    /// Repeats array elements.
    pub fn repeat(&self, repeats: usize, axis: Option<usize>) -> Self {
        match axis {
            None => {
                let mut data = Vec::with_capacity(self.data.len() * repeats);
                for &value in &self.data {
                    for _ in 0..repeats {
                        data.push(value);
                    }
                }
                Self::from_shape_vec(&[data.len()], data)
            }
            Some(axis) => {
                assert!(axis < self.ndim(), "axis {axis} out of bounds");
                let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);
                let mut shape = self.shape.clone();
                shape[axis] *= repeats;
                let mut data = Vec::with_capacity(compute_size(&shape));

                for outer_index in 0..outer {
                    for axis_index in 0..axis_len {
                        let start = (outer_index * axis_len + axis_index) * inner;
                        let end = start + inner;
                        for _ in 0..repeats {
                            data.extend_from_slice(&self.data[start..end]);
                        }
                    }
                }

                Self::from_shape_vec(&shape, data)
            }
        }
    }

    /// Tiles an array across each axis.
    pub fn tile(&self, reps: &[usize]) -> Self {
        if self.ndim() == 2 && reps.len() == 2 {
            let row_repeat = reps[0];
            let col_repeat = reps[1];
            if row_repeat == 1 && col_repeat == 1 {
                return self.copy();
            }

            let rows = self.shape[0];
            let cols = self.shape[1];
            let target_rows = rows * row_repeat;
            let target_cols = cols * col_repeat;
            let mut data = vec![0.0; target_rows * target_cols];
            let mut expanded_row = vec![0.0; target_cols];

            for src_row in 0..rows {
                let source = &self.data[src_row * cols..(src_row + 1) * cols];
                for repeat_index in 0..col_repeat {
                    let start = repeat_index * cols;
                    expanded_row[start..start + cols].copy_from_slice(source);
                }

                for repeat_block in 0..row_repeat {
                    let out_row = repeat_block * rows + src_row;
                    let output_row = &mut data[out_row * target_cols..(out_row + 1) * target_cols];
                    output_row.copy_from_slice(&expanded_row);
                }
            }

            return Self::from_shape_vec(&[target_rows, target_cols], data);
        }

        let ndim = self.ndim().max(reps.len());
        let mut source_shape = SmallVec::<[usize; 6]>::from_elem(1, ndim - self.ndim());
        source_shape.extend_from_slice(&self.shape);

        let mut aligned_reps = SmallVec::<[usize; 6]>::from_elem(1, ndim - reps.len());
        aligned_reps.extend_from_slice(reps);

        let target_shape = source_shape
            .iter()
            .zip(aligned_reps.iter())
            .map(|(&dimension, &repeat)| dimension * repeat)
            .collect::<SmallVec<[usize; 6]>>();
        let target_strides = compute_strides(&target_shape);
        let size = compute_size(&target_shape);

        if source_shape == target_shape {
                return Self {
                    data: clone_data_parallel(&self.data),
                    shape: target_shape,
                    strides: target_strides,
                };
        }

        let mut data = clone_data_parallel(&self.data);
        let mut current_shape = source_shape.clone();

        for axis in 0..ndim {
            let repeat = aligned_reps[axis];
            if repeat <= 1 {
                continue;
            }

            let (inner, outer, axis_len) = axis_inner_outer(&current_shape, axis);
            let block_len = axis_len * inner;
            let mut tiled = Vec::with_capacity(data.len() * repeat);

            for outer_index in 0..outer {
                let start = outer_index * block_len;
                let end = start + block_len;
                for _ in 0..repeat {
                    tiled.extend_from_slice(&data[start..end]);
                }
            }

            data = tiled;
            current_shape[axis] *= repeat;
        }

        debug_assert_eq!(data.len(), size);
        Self::from_shape_vec(&target_shape, data)
    }
}
