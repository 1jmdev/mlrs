use crate::darray::Array;

use super::super::PreprocessingError;
use super::super::common::{
    checked_feature_range, column_min_max, ensure_2d_finite, ensure_feature_count,
    is_effectively_zero,
};

/// Scales each feature into a user-provided interval.
#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScaler {
    /// Sets the output interval applied to every feature.
    pub feature_range: (f64, f64),
    /// Controls whether transformed values are clipped to the output interval.
    pub clip: bool,
    /// Stores per-feature offsets used during affine transformation.
    pub min_: Option<Array>,
    /// Stores per-feature scaling factors.
    pub scale_: Option<Array>,
    /// Stores observed per-feature minimum values.
    pub data_min_: Option<Array>,
    /// Stores observed per-feature maximum values.
    pub data_max_: Option<Array>,
    /// Stores observed per-feature ranges.
    pub data_range_: Option<Array>,
    /// Stores the number of features observed during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for MinMaxScaler {
    /// Returns the default min-max scaling configuration.
    fn default() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            clip: false,
            min_: None,
            scale_: None,
            data_min_: None,
            data_max_: None,
            data_range_: None,
            n_features_in_: None,
        }
    }
}

impl MinMaxScaler {
    /// Creates a min-max scaler with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific output interval.
    pub fn feature_range(mut self, feature_range: (f64, f64)) -> Self {
        self.feature_range = feature_range;
        self
    }

    /// Returns a copy with a specific clipping policy.
    pub fn clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Reports whether fitted affine parameters are available.
    pub fn is_fitted(&self) -> bool {
        self.min_.is_some() && self.scale_.is_some() && self.n_features_in_.is_some()
    }

    /// Learns per-feature affine scaling parameters.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        let (lower, upper) = checked_feature_range(self.feature_range.0, self.feature_range.1)?;
        let (data_min, data_max) = column_min_max(x, rows, cols);
        let data_range = data_max
            .iter()
            .zip(&data_min)
            .map(|(max, min)| max - min)
            .collect::<Vec<_>>();
        let scale = data_range
            .iter()
            .map(|range| {
                let divisor = if is_effectively_zero(*range) {
                    1.0
                } else {
                    *range
                };
                (upper - lower) / divisor
            })
            .collect::<Vec<_>>();
        let min = data_min
            .iter()
            .zip(&scale)
            .map(|(data_min, scale)| lower - data_min * scale)
            .collect::<Vec<_>>();

        self.min_ = Some(Array::from_shape_vec(&[cols], min));
        self.scale_ = Some(Array::from_shape_vec(&[cols], scale));
        self.data_min_ = Some(Array::from_shape_vec(&[cols], data_min));
        self.data_max_ = Some(Array::from_shape_vec(&[cols], data_max));
        self.data_range_ = Some(Array::from_shape_vec(&[cols], data_range));
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Fits the scaler and returns the transformed matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Applies the fitted affine scaling transform.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("MinMaxScaler"))?,
        )?;

        let min = self
            .min_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("MinMaxScaler"))?
            .data();
        let scale = self
            .scale_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("MinMaxScaler"))?
            .data();
        let (lower, upper) = self.feature_range;
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                let mut value = x.data()[offset + col] * scale[col] + min[col];
                if self.clip {
                    value = value.clamp(lower, upper);
                }
                data.push(value);
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }

    /// Maps scaled features back into their original interval.
    pub fn inverse_transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("MinMaxScaler"))?,
        )?;

        let min = self
            .min_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("MinMaxScaler"))?
            .data();
        let scale = self
            .scale_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("MinMaxScaler"))?
            .data();
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                data.push((x.data()[offset + col] - min[col]) / scale[col]);
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
