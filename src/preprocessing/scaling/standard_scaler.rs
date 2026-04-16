use crate::darray::Array;

use super::super::PreprocessingError;
use super::super::common::{
    SIMD_LANES, column_mean_var, ensure_2d_finite, ensure_feature_count, is_effectively_zero,
    load_f64x4, store_f64x4,
};

/// Standardizes features by removing the mean and scaling to unit variance.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardScaler {
    /// Controls whether each feature is centered before scaling.
    pub with_mean: bool,
    /// Controls whether each feature is scaled to unit variance.
    pub with_std: bool,
    /// Stores fitted per-feature means when required by the configuration.
    pub mean_: Option<Array>,
    /// Stores fitted per-feature population variances when standardization is enabled.
    pub var_: Option<Array>,
    /// Stores fitted per-feature scaling factors.
    pub scale_: Option<Array>,
    /// Stores the number of features observed during fitting.
    pub n_features_in_: Option<usize>,
    /// Stores the number of samples observed during fitting.
    pub n_samples_seen_: Option<usize>,
}

impl Default for StandardScaler {
    /// Returns the sklearn-compatible default configuration.
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            mean_: None,
            var_: None,
            scale_: None,
            n_features_in_: None,
            n_samples_seen_: None,
        }
    }
}

impl StandardScaler {
    /// Creates a standard scaler with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific centering policy.
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Returns a copy with a specific variance scaling policy.
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }

    /// Reports whether the scaler has learned all required statistics.
    pub fn is_fitted(&self) -> bool {
        self.n_features_in_.is_some()
            && self.n_samples_seen_.is_some()
            && (!self.with_mean && !self.with_std || self.mean_.is_some())
            && (!self.with_std || (self.var_.is_some() && self.scale_.is_some()))
    }

    /// Learns per-feature statistics from a feature matrix.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        let (means, variances) = column_mean_var(x, rows, cols);
        let scales = variances
            .iter()
            .map(|variance| {
                let std = variance.sqrt();
                if is_effectively_zero(std) { 1.0 } else { std }
            })
            .collect::<Vec<_>>();

        self.mean_ =
            (self.with_mean || self.with_std).then(|| Array::from_shape_vec(&[cols], means));
        self.var_ = self
            .with_std
            .then(|| Array::from_shape_vec(&[cols], variances));
        self.scale_ = self
            .with_std
            .then(|| Array::from_shape_vec(&[cols], scales));
        self.n_features_in_ = Some(cols);
        self.n_samples_seen_ = Some(rows);
        Ok(self)
    }

    /// Fits the scaler and returns the transformed matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Applies the fitted centering and scaling parameters.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("StandardScaler"))?,
        )?;

        let means = self.mean_.as_ref().map(Array::data);
        let scales = self.scale_.as_ref().map(Array::data);
        let input = x.data();
        let mut data = vec![0.0; rows * cols];

        for row in 0..rows {
            let offset = row * cols;
            let src = &input[offset..offset + cols];
            let dst = &mut data[offset..offset + cols];
            let mut col = 0;

            match (self.with_mean, self.with_std) {
                (true, true) => {
                    let means = means.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    let scales = scales.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    while col + SIMD_LANES <= cols {
                        let values = (load_f64x4(src, col) - load_f64x4(means, col))
                            / load_f64x4(scales, col);
                        store_f64x4(dst, col, values);
                        col += SIMD_LANES;
                    }
                    while col < cols {
                        dst[col] = (src[col] - means[col]) / scales[col];
                        col += 1;
                    }
                }
                (true, false) => {
                    let means = means.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    while col + SIMD_LANES <= cols {
                        let values = load_f64x4(src, col) - load_f64x4(means, col);
                        store_f64x4(dst, col, values);
                        col += SIMD_LANES;
                    }
                    while col < cols {
                        dst[col] = src[col] - means[col];
                        col += 1;
                    }
                }
                (false, true) => {
                    let scales = scales.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    while col + SIMD_LANES <= cols {
                        let values = load_f64x4(src, col) / load_f64x4(scales, col);
                        store_f64x4(dst, col, values);
                        col += SIMD_LANES;
                    }
                    while col < cols {
                        dst[col] = src[col] / scales[col];
                        col += 1;
                    }
                }
                (false, false) => dst.copy_from_slice(src),
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }

    /// Maps standardized data back into the original feature space.
    pub fn inverse_transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("StandardScaler"))?,
        )?;

        let means = self.mean_.as_ref().map(Array::data);
        let scales = self.scale_.as_ref().map(Array::data);
        let input = x.data();
        let mut data = vec![0.0; rows * cols];

        for row in 0..rows {
            let offset = row * cols;
            let src = &input[offset..offset + cols];
            let dst = &mut data[offset..offset + cols];
            let mut col = 0;

            match (self.with_mean, self.with_std) {
                (true, true) => {
                    let means = means.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    let scales = scales.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    while col + SIMD_LANES <= cols {
                        let values =
                            load_f64x4(src, col) * load_f64x4(scales, col) + load_f64x4(means, col);
                        store_f64x4(dst, col, values);
                        col += SIMD_LANES;
                    }
                    while col < cols {
                        dst[col] = src[col] * scales[col] + means[col];
                        col += 1;
                    }
                }
                (true, false) => {
                    let means = means.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    while col + SIMD_LANES <= cols {
                        let values = load_f64x4(src, col) + load_f64x4(means, col);
                        store_f64x4(dst, col, values);
                        col += SIMD_LANES;
                    }
                    while col < cols {
                        dst[col] = src[col] + means[col];
                        col += 1;
                    }
                }
                (false, true) => {
                    let scales = scales.ok_or(PreprocessingError::NotFitted("StandardScaler"))?;
                    while col + SIMD_LANES <= cols {
                        let values = load_f64x4(src, col) * load_f64x4(scales, col);
                        store_f64x4(dst, col, values);
                        col += SIMD_LANES;
                    }
                    while col < cols {
                        dst[col] = src[col] * scales[col];
                        col += 1;
                    }
                }
                (false, false) => dst.copy_from_slice(src),
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
