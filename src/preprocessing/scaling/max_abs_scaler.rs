use crate::darray::Array;

use rayon::prelude::*;

use super::super::common::{
    ensure_2d_finite, ensure_feature_count, is_effectively_zero, load_f64x4, store_f64x4,
    SIMD_LANES,
};
use super::super::PreprocessingError;

const PARALLEL_THRESHOLD: usize = 16_384;

/// Scales each feature by its maximum absolute training value.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxAbsScaler {
    /// Stores fitted per-feature absolute maxima.
    pub max_abs_: Option<Array>,
    /// Stores fitted per-feature scaling factors.
    pub scale_: Option<Array>,
    /// Stores the number of features observed during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for MaxAbsScaler {
    /// Returns the default max-absolute scaling configuration.
    fn default() -> Self {
        Self {
            max_abs_: None,
            scale_: None,
            n_features_in_: None,
        }
    }
}

impl MaxAbsScaler {
    /// Creates a max-absolute scaler with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reports whether the scaler has learned per-feature maxima.
    pub fn is_fitted(&self) -> bool {
        self.max_abs_.is_some() && self.scale_.is_some() && self.n_features_in_.is_some()
    }

    /// Learns per-feature absolute maxima from a feature matrix.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        let data = x.data();
        let max_abs = if rows * cols >= PARALLEL_THRESHOLD {
            data.par_chunks(cols)
                .fold(
                    || vec![0.0; cols],
                    |mut local_max, row_values| {
                        update_max_abs_row(&mut local_max, row_values);
                        local_max
                    },
                )
                .reduce(
                    || vec![0.0; cols],
                    |mut left, right| {
                        merge_max_abs(&mut left, &right);
                        left
                    },
                )
        } else {
            let mut max_abs = vec![0.0; cols];
            for row_values in data.chunks_exact(cols) {
                update_max_abs_row(&mut max_abs, row_values);
            }
            max_abs
        };

        let scale = max_abs
            .iter()
            .map(|value| {
                if is_effectively_zero(*value) {
                    1.0
                } else {
                    *value
                }
            })
            .collect::<Vec<_>>();

        self.max_abs_ = Some(Array::from_shape_vec(&[cols], max_abs));
        self.scale_ = Some(Array::from_shape_vec(&[cols], scale));
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Fits the scaler and returns the transformed matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Divides each feature by its fitted maximum absolute value.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("MaxAbsScaler"))?,
        )?;

        let scales = self
            .scale_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("MaxAbsScaler"))?
            .data();
        let input = x.data();
        let mut data = vec![0.0; rows * cols];
        if rows * cols >= PARALLEL_THRESHOLD {
            data.par_chunks_mut(cols)
                .zip(input.par_chunks(cols))
                .for_each(|(dst, src)| divide_row(dst, src, scales));
        } else {
            for (dst, src) in data.chunks_exact_mut(cols).zip(input.chunks_exact(cols)) {
                divide_row(dst, src, scales);
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }

    /// Restores max-absolute scaled features to the original scale.
    pub fn inverse_transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("MaxAbsScaler"))?,
        )?;

        let scales = self
            .scale_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("MaxAbsScaler"))?
            .data();
        let input = x.data();
        let mut data = vec![0.0; rows * cols];
        if rows * cols >= PARALLEL_THRESHOLD {
            data.par_chunks_mut(cols)
                .zip(input.par_chunks(cols))
                .for_each(|(dst, src)| multiply_row(dst, src, scales));
        } else {
            for (dst, src) in data.chunks_exact_mut(cols).zip(input.chunks_exact(cols)) {
                multiply_row(dst, src, scales);
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}

fn update_max_abs_row(max_abs: &mut [f64], row_values: &[f64]) {
    let mut col = 0;
    while col + SIMD_LANES <= row_values.len() {
        let values = load_f64x4(row_values, col).abs();
        let current = load_f64x4(max_abs, col);
        store_f64x4(max_abs, col, current.max(values));
        col += SIMD_LANES;
    }
    while col < row_values.len() {
        let candidate = row_values[col].abs();
        if candidate > max_abs[col] {
            max_abs[col] = candidate;
        }
        col += 1;
    }
}

fn merge_max_abs(left: &mut [f64], right: &[f64]) {
    let mut col = 0;
    while col + SIMD_LANES <= left.len() {
        let merged = load_f64x4(left, col).max(load_f64x4(right, col));
        store_f64x4(left, col, merged);
        col += SIMD_LANES;
    }
    while col < left.len() {
        left[col] = left[col].max(right[col]);
        col += 1;
    }
}

fn divide_row(dst: &mut [f64], src: &[f64], scales: &[f64]) {
    let mut col = 0;
    while col + SIMD_LANES <= src.len() {
        store_f64x4(dst, col, load_f64x4(src, col) / load_f64x4(scales, col));
        col += SIMD_LANES;
    }
    while col < src.len() {
        dst[col] = src[col] / scales[col];
        col += 1;
    }
}

fn multiply_row(dst: &mut [f64], src: &[f64], scales: &[f64]) {
    let mut col = 0;
    while col + SIMD_LANES <= src.len() {
        store_f64x4(dst, col, load_f64x4(src, col) * load_f64x4(scales, col));
        col += SIMD_LANES;
    }
    while col < src.len() {
        dst[col] = src[col] * scales[col];
        col += 1;
    }
}
