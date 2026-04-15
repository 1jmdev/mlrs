use super::super::Array;
use super::super::utils::{axis_inner_outer, compute_size, compute_strides};
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
        let source_strides = compute_strides(&source_shape);
        let size = compute_size(&target_shape);

        if source_shape == target_shape {
            return Self {
                data: self.data.clone(),
                shape: target_shape,
                strides: target_strides,
            };
        }

        if ndim == 1 {
            let mut data = Vec::with_capacity(size);
            for _ in 0..aligned_reps[0] {
                data.extend_from_slice(&self.data);
            }
            return Self::from_shape_vec(&target_shape, data);
        }

        let mut data = Vec::with_capacity(size);

        for flat_index in 0..size {
            let mut remainder = flat_index;
            let mut source_offset = 0;

            for axis in 0..ndim {
                let coordinate = if target_shape.is_empty() {
                    0
                } else {
                    remainder / target_strides[axis]
                };
                if !target_shape.is_empty() {
                    remainder %= target_strides[axis];
                }
                let source_coordinate = coordinate % source_shape[axis];
                source_offset += source_coordinate * source_strides[axis];
            }

            data.push(self.data[source_offset]);
        }

        Self::from_shape_vec(&target_shape, data)
    }
}
