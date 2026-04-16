use super::super::Array;
use super::super::utils::{PAR_THRESHOLD, axis_inner_outer};
use rayon::prelude::*;

impl Array {
    pub fn sort(&self) -> Self {
        let mut data = self.to_vec();
        if data.len() >= PAR_THRESHOLD {
            data.par_sort_by(f64::total_cmp);
        } else {
            data.sort_by(f64::total_cmp);
        }
        Self::from_shape_vec(&[data.len()], data)
    }

    pub fn argsort(&self) -> Self {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        if indices.len() >= PAR_THRESHOLD {
            indices
                .par_sort_unstable_by(|&left, &right| self.data[left].total_cmp(&self.data[right]));
        } else {
            indices.sort_unstable_by(|&left, &right| self.data[left].total_cmp(&self.data[right]));
        }
        let data = indices.into_iter().map(|index| index as f64).collect();
        Self::from_shape_vec(&[self.len()], data)
    }

    pub fn sort_axis(&self, axis: usize) -> Self {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");
        let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);
        let mut data = self.to_vec();

        if inner == 1 {
            for outer_index in 0..outer {
                let start = outer_index * axis_len;
                let end = start + axis_len;
                data[start..end].sort_by(f64::total_cmp);
            }
            return Self::from_shape_vec(&self.shape, data);
        }

        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let mut values = Vec::with_capacity(axis_len);
                for axis_index in 0..axis_len {
                    values.push(
                        self.data[(outer_index * axis_len + axis_index) * inner + inner_index],
                    );
                }
                values.sort_by(f64::total_cmp);
                for (axis_index, value) in values.into_iter().enumerate() {
                    let offset = (outer_index * axis_len + axis_index) * inner + inner_index;
                    data[offset] = value;
                }
            }
        }

        Self::from_shape_vec(&self.shape, data)
    }

    pub fn argsort_axis(&self, axis: usize) -> Self {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");
        let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);
        let mut data = vec![0.0; self.len()];
        let mut pairs = vec![(0.0, 0usize); axis_len];

        for outer_index in 0..outer {
            for inner_index in 0..inner {
                for axis_index in 0..axis_len {
                    let offset = (outer_index * axis_len + axis_index) * inner + inner_index;
                    pairs[axis_index] = (self.data[offset], axis_index);
                }

                pairs.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

                for (axis_index, &(_, index)) in pairs.iter().enumerate() {
                    let offset = (outer_index * axis_len + axis_index) * inner + inner_index;
                    data[offset] = index as f64;
                }
            }
        }

        Self::from_shape_vec(&self.shape, data)
    }
}
