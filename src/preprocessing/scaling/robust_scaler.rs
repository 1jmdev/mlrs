use crate::darray::Array;

use super::super::PreprocessingError;
use super::super::common::{
    checked_quantile_range, column_percentile, ensure_2d_finite, ensure_feature_count,
    is_effectively_zero,
};

/// Scales features using statistics that are robust to outliers.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustScaler {
    /// Controls whether each feature is centered by its median.
    pub with_centering: bool,
    /// Controls whether each feature is scaled by its inter-quantile range.
    pub with_scaling: bool,
    /// Sets the lower and upper quantiles, expressed as percentages.
    pub quantile_range: (f64, f64),
    /// Stores fitted per-feature medians when centering is enabled.
    pub center_: Option<Array>,
    /// Stores fitted per-feature robust scaling factors.
    pub scale_: Option<Array>,
    /// Stores the number of features observed during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for RobustScaler {
    /// Returns the default robust scaling configuration.
    fn default() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
            center_: None,
            scale_: None,
            n_features_in_: None,
        }
    }
}

impl RobustScaler {
    /// Creates a robust scaler with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific centering policy.
    pub fn with_centering(mut self, with_centering: bool) -> Self {
        self.with_centering = with_centering;
        self
    }

    /// Returns a copy with a specific scaling policy.
    pub fn with_scaling(mut self, with_scaling: bool) -> Self {
        self.with_scaling = with_scaling;
        self
    }

    /// Returns a copy with a specific quantile interval.
    pub fn quantile_range(mut self, quantile_range: (f64, f64)) -> Self {
        self.quantile_range = quantile_range;
        self
    }

    /// Reports whether fitted robust statistics are available.
    pub fn is_fitted(&self) -> bool {
        self.n_features_in_.is_some()
            && (!self.with_centering || self.center_.is_some())
            && (!self.with_scaling || self.scale_.is_some())
    }

    /// Learns medians and inter-quantile scales from a feature matrix.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        let (lower, upper) = checked_quantile_range(self.quantile_range.0, self.quantile_range.1)?;
        let medians = column_percentile(x, rows, cols, 0.5);
        let lowers = column_percentile(x, rows, cols, lower / 100.0);
        let uppers = column_percentile(x, rows, cols, upper / 100.0);
        let scales = lowers
            .iter()
            .zip(&uppers)
            .map(|(lower, upper)| {
                let spread = upper - lower;
                if is_effectively_zero(spread) {
                    1.0
                } else {
                    spread
                }
            })
            .collect::<Vec<_>>();

        self.center_ = self
            .with_centering
            .then(|| Array::from_shape_vec(&[cols], medians));
        self.scale_ = self
            .with_scaling
            .then(|| Array::from_shape_vec(&[cols], scales));
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Fits the scaler and returns the transformed matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Applies the fitted robust centering and scaling transform.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("RobustScaler"))?,
        )?;

        let centers = self.center_.as_ref().map(Array::data);
        let scales = self.scale_.as_ref().map(Array::data);
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                let mut value = x.data()[offset + col];
                if self.with_centering {
                    value -= centers.ok_or(PreprocessingError::NotFitted("RobustScaler"))?[col];
                }
                if self.with_scaling {
                    value /= scales.ok_or(PreprocessingError::NotFitted("RobustScaler"))?[col];
                }
                data.push(value);
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }

    /// Maps robust-scaled features back into the original feature space.
    pub fn inverse_transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("RobustScaler"))?,
        )?;

        let centers = self.center_.as_ref().map(Array::data);
        let scales = self.scale_.as_ref().map(Array::data);
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                let mut value = x.data()[offset + col];
                if self.with_scaling {
                    value *= scales.ok_or(PreprocessingError::NotFitted("RobustScaler"))?[col];
                }
                if self.with_centering {
                    value += centers.ok_or(PreprocessingError::NotFitted("RobustScaler"))?[col];
                }
                data.push(value);
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
