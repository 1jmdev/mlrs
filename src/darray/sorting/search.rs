use super::super::Array;
use super::super::utils::PAR_THRESHOLD;
use super::SearchSide;
use rayon::prelude::*;

impl Array {
    pub fn searchsorted(&self, values: &Self, side: SearchSide) -> Self {
        assert_eq!(self.ndim(), 1, "searchsorted() requires a 1-D sorted array");
        let data = if values.len() >= PAR_THRESHOLD {
            values
                .data()
                .par_iter()
                .map(|&value| self.searchsorted_index(value, side) as f64)
                .collect()
        } else {
            values
                .data()
                .iter()
                .map(|&value| self.searchsorted_index(value, side) as f64)
                .collect()
        };
        Self::from_shape_vec(values.shape(), data)
    }

    fn searchsorted_index(&self, value: f64, side: SearchSide) -> usize {
        match side {
            SearchSide::Left => self.data.partition_point(|entry| *entry < value),
            SearchSide::Right => self.data.partition_point(|entry| *entry <= value),
        }
    }
}
