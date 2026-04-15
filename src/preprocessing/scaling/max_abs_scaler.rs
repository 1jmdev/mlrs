use crate::darray::Array;

use super::super::common::{ensure_2d_finite, ensure_feature_count, is_effectively_zero};
use super::super::PreprocessingError;

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
        let mut max_abs = vec![0.0; cols];

        for row in 0..rows {
            let offset = row * cols;
            for (col, current_max) in max_abs.iter_mut().enumerate() {
                let candidate = x.data()[offset + col].abs();
                if candidate > *current_max {
                    *current_max = candidate;
                }
            }
        }

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
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let offset = row * cols;
            data.extend((0..cols).map(|col| x.data()[offset + col] / scales[col]));
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
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let offset = row * cols;
            data.extend((0..cols).map(|col| x.data()[offset + col] * scales[col]));
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
