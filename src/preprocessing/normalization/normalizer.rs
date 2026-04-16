use crate::darray::Array;

use super::super::PreprocessingError;
use super::super::common::{
    SIMD_LANES, ensure_2d_finite, ensure_feature_count, is_effectively_zero, load_f64x4,
    reduce_max, reduce_sum, store_f64x4,
};
use super::Norm;
use wide::f64x4;

/// Normalizes each sample independently to unit norm.
#[derive(Debug, Clone, PartialEq)]
pub struct Normalizer {
    /// Selects the norm applied to each sample row.
    pub norm: Norm,
    /// Stores the number of features validated during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for Normalizer {
    /// Returns the default L2 normalizer.
    fn default() -> Self {
        Self {
            norm: Norm::L2,
            n_features_in_: None,
        }
    }
}

impl Normalizer {
    /// Creates a normalizer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific row norm.
    pub fn norm(mut self, norm: Norm) -> Self {
        self.norm = norm;
        self
    }

    /// Reports whether the normalizer has validated a feature count.
    pub fn is_fitted(&self) -> bool {
        self.n_features_in_.is_some()
    }

    /// Validates the feature matrix shape and stores the feature count.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (_, cols) = ensure_2d_finite(x, "X")?;
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Validates and immediately normalizes a feature matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Normalizes each sample row independently.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("Normalizer"))?,
        )?;

        let input = x.data();
        let mut data = vec![0.0; rows * cols];
        for row in 0..rows {
            let offset = row * cols;
            let row_values = &input[offset..offset + cols];
            let row_output = &mut data[offset..offset + cols];
            let norm = match self.norm {
                Norm::L1 => {
                    let mut accum = f64x4::from([0.0; SIMD_LANES]);
                    let mut col = 0;
                    while col + SIMD_LANES <= cols {
                        accum += load_f64x4(row_values, col).abs();
                        col += SIMD_LANES;
                    }
                    let mut total = reduce_sum(accum);
                    while col < cols {
                        total += row_values[col].abs();
                        col += 1;
                    }
                    total
                }
                Norm::L2 => {
                    let mut accum = f64x4::from([0.0; SIMD_LANES]);
                    let mut col = 0;
                    while col + SIMD_LANES <= cols {
                        let values = load_f64x4(row_values, col);
                        accum += values * values;
                        col += SIMD_LANES;
                    }
                    let mut total = reduce_sum(accum);
                    while col < cols {
                        total += row_values[col] * row_values[col];
                        col += 1;
                    }
                    total.sqrt()
                }
                Norm::Max => {
                    let mut accum = f64x4::from([0.0; SIMD_LANES]);
                    let mut col = 0;
                    while col + SIMD_LANES <= cols {
                        accum = accum.max(load_f64x4(row_values, col).abs());
                        col += SIMD_LANES;
                    }
                    let mut total = reduce_max(accum);
                    while col < cols {
                        total = total.max(row_values[col].abs());
                        col += 1;
                    }
                    total
                }
            };

            if is_effectively_zero(norm) {
                row_output.copy_from_slice(row_values);
                continue;
            }

            let norm_vector = f64x4::from([norm; SIMD_LANES]);
            let mut col = 0;
            while col + SIMD_LANES <= cols {
                store_f64x4(row_output, col, load_f64x4(row_values, col) / norm_vector);
                col += SIMD_LANES;
            }
            while col < cols {
                row_output[col] = row_values[col] / norm;
                col += 1;
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
