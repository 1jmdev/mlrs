use crate::darray::Array;

use super::super::PreprocessingError;
use super::super::common::{ensure_2d_finite, ensure_feature_count, is_effectively_zero};
use super::Norm;

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

        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let offset = row * cols;
            let row_values = &x.data()[offset..offset + cols];
            let norm = match self.norm {
                Norm::L1 => row_values.iter().map(|value| value.abs()).sum::<f64>(),
                Norm::L2 => row_values
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt(),
                Norm::Max => row_values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f64::max),
            };

            if is_effectively_zero(norm) {
                data.extend_from_slice(row_values);
                continue;
            }

            data.extend(row_values.iter().map(|value| value / norm));
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
