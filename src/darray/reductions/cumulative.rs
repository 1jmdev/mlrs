use super::super::Array;
use super::super::utils::{PAR_THRESHOLD, axis_inner_outer};
use rayon::prelude::*;

impl Array {
    /// Returns the cumulative sum over the flattened array.
    pub fn cumsum(&self) -> Self {
        let mut running = 0.0;
        let data = self
            .data
            .iter()
            .map(|&value| {
                running += value;
                running
            })
            .collect::<Vec<_>>();
        Self::from_shape_vec(&self.shape, data)
    }

    /// Returns the cumulative product over the flattened array.
    pub fn cumprod(&self) -> Self {
        let mut running = 1.0;
        let data = self
            .data
            .iter()
            .map(|&value| {
                running *= value;
                running
            })
            .collect::<Vec<_>>();
        Self::from_shape_vec(&self.shape, data)
    }

    /// Returns the cumulative sum along an axis.
    pub fn cumsum_axis(&self, axis: usize) -> Self {
        self.accumulate_axis(axis, 0.0, |running, value| *running += value)
    }

    /// Returns the cumulative product along an axis.
    pub fn cumprod_axis(&self, axis: usize) -> Self {
        self.accumulate_axis(axis, 1.0, |running, value| *running *= value)
    }

    fn accumulate_axis<F>(&self, axis: usize, init: f64, accumulate: F) -> Self
    where
        F: Fn(&mut f64, f64) + Copy + Send + Sync,
    {
        assert!(axis < self.ndim(), "axis {axis} out of bounds");
        let (inner, outer, axis_len) = axis_inner_outer(&self.shape, axis);
        let mut data = self.data.clone();

        if inner == 1 {
            if outer >= PAR_THRESHOLD {
                data.par_chunks_mut(axis_len).for_each(|chunk| {
                    let mut running = init;
                    for value in chunk {
                        accumulate(&mut running, *value);
                        *value = running;
                    }
                });
            } else {
                for chunk in data.chunks_mut(axis_len) {
                    let mut running = init;
                    for value in chunk {
                        accumulate(&mut running, *value);
                        *value = running;
                    }
                }
            }
            return Self::from_shape_vec(&self.shape, data);
        }

        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let mut running = init;
                for axis_index in 0..axis_len {
                    let offset = (outer_index * axis_len + axis_index) * inner + inner_index;
                    accumulate(&mut running, self.data[offset]);
                    data[offset] = running;
                }
            }
        }

        Self::from_shape_vec(&self.shape, data)
    }
}
