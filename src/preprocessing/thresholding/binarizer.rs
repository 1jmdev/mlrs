use crate::darray::Array;

use super::super::common::{ensure_2d_finite, ensure_feature_count};
use super::super::PreprocessingError;

/// Thresholds each feature value into a binary output.
#[derive(Debug, Clone, PartialEq)]
pub struct Binarizer {
    /// Values strictly greater than this threshold map to one.
    pub threshold: f64,
    /// Stores the number of features validated during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for Binarizer {
    /// Returns the default thresholding configuration.
    fn default() -> Self {
        Self {
            threshold: 0.0,
            n_features_in_: None,
        }
    }
}

impl Binarizer {
    /// Creates a binarizer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific threshold.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Reports whether the binarizer has validated a feature count.
    pub fn is_fitted(&self) -> bool {
        self.n_features_in_.is_some()
    }

    /// Validates the feature matrix shape and stores the feature count.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (_, cols) = ensure_2d_finite(x, "X")?;
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Validates and immediately binarizes a feature matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Maps each value to zero or one using the fitted threshold.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("Binarizer"))?,
        )?;

        let data = x
            .data()
            .iter()
            .map(|value| if *value > self.threshold { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
