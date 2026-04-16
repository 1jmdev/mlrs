use super::super::Array;
use super::super::utils::{PAR_THRESHOLD, axis_inner_outer, reduce_axis, reduced_shape, sum_simd};
use rayon::prelude::*;

impl Array {
    /// Reduces an axis by summing along it.
    pub fn sum_axis(&self, axis: usize) -> Self {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");
        let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);
        let shape = reduced_shape(&self.shape, axis);

        if inner == 1 {
            let mut data = vec![0.0; outer];
            if outer >= PAR_THRESHOLD {
                data.par_iter_mut()
                    .enumerate()
                    .for_each(|(outer_index, value)| {
                        let start = outer_index * axis_len;
                        let end = start + axis_len;
                        *value = sum_simd(&self.data[start..end]);
                    });
            } else {
                for (outer_index, value) in data.iter_mut().enumerate() {
                    let start = outer_index * axis_len;
                    let end = start + axis_len;
                    *value = sum_simd(&self.data[start..end]);
                }
            }
            return Self::from_shape_vec(&shape, data);
        }

        reduce_axis(self, axis, 0.0, |accumulator, value| accumulator + value)
    }

    /// Reduces an axis by multiplying along it.
    pub fn prod_axis(&self, axis: usize) -> Self {
        reduce_axis(self, axis, 1.0, |accumulator, value| accumulator * value)
    }

    /// Reduces an axis by taking its minimum.
    pub fn min_axis(&self, axis: usize) -> Self {
        reduce_axis(self, axis, f64::INFINITY, |accumulator, value| {
            accumulator.min(value)
        })
    }

    /// Reduces an axis by taking its maximum.
    pub fn max_axis(&self, axis: usize) -> Self {
        reduce_axis(self, axis, f64::NEG_INFINITY, |accumulator, value| {
            accumulator.max(value)
        })
    }

    /// Reduces an axis by computing its mean.
    pub fn mean_axis(&self, axis: usize) -> Self {
        let sums = self.sum_axis(axis);
        let divisor = self.shape[axis] as f64;
        sums.scale(1.0 / divisor)
    }

    /// Returns the population variance along an axis.
    pub fn var_axis(&self, axis: usize) -> Self {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");
        let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);
        let shape = reduced_shape(&self.shape, axis);
        let divisor = axis_len as f64;
        let out_len = outer * inner;
        let mut data = vec![0.0; out_len];

        if out_len >= PAR_THRESHOLD {
            data.par_iter_mut().enumerate().for_each(|(index, value)| {
                let outer_index = index / inner;
                let inner_index = index % inner;
                let base = outer_index * axis_len * inner + inner_index;

                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                for axis_index in 0..axis_len {
                    let current = self.data[base + axis_index * inner];
                    sum += current;
                    sum_sq += current * current;
                }

                let mean = sum / divisor;
                *value = (sum_sq / divisor) - mean * mean;
            });
        } else {
            for (index, value) in data.iter_mut().enumerate() {
                let outer_index = index / inner;
                let inner_index = index % inner;
                let base = outer_index * axis_len * inner + inner_index;

                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                for axis_index in 0..axis_len {
                    let current = self.data[base + axis_index * inner];
                    sum += current;
                    sum_sq += current * current;
                }

                let mean = sum / divisor;
                *value = (sum_sq / divisor) - mean * mean;
            }
        }

        Self::from_shape_vec(&shape, data)
    }

    /// Returns the population standard deviation along an axis.
    pub fn std_axis(&self, axis: usize) -> Self {
        self.var_axis(axis).sqrt()
    }

    /// Returns whether all elements along an axis are non-zero.
    pub fn all_axis(&self, axis: usize) -> Self {
        reduce_axis(self, axis, 1.0, |accumulator, value| {
            if accumulator != 0.0 && value != 0.0 {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Returns whether any element along an axis is non-zero.
    pub fn any_axis(&self, axis: usize) -> Self {
        reduce_axis(self, axis, 0.0, |accumulator, value| {
            if accumulator != 0.0 || value != 0.0 {
                1.0
            } else {
                0.0
            }
        })
    }
}
