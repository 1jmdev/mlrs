use crate::darray::Array;

use super::super::common::{ensure_2d_finite, ensure_feature_count};
use super::super::PreprocessingError;
use super::combinations::generate_powers;

/// Expands features into a polynomial basis up to a fixed degree.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialFeatures {
    /// Sets the maximum polynomial degree included in the expansion.
    pub degree: usize,
    /// Restricts generated terms to products of distinct input features.
    pub interaction_only: bool,
    /// Prepends a bias column of ones to the expansion.
    pub include_bias: bool,
    /// Stores the input feature count observed during fitting.
    pub n_features_in_: Option<usize>,
    /// Stores the number of generated output features.
    pub n_output_features_: Option<usize>,
    /// Stores the exponent vector for every generated output feature.
    pub powers_: Option<Array>,
}

impl Default for PolynomialFeatures {
    /// Returns the default polynomial feature expansion configuration.
    fn default() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            n_features_in_: None,
            n_output_features_: None,
            powers_: None,
        }
    }
}

impl PolynomialFeatures {
    /// Creates a polynomial feature generator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific maximum degree.
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Returns a copy with a specific interaction-only policy.
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Returns a copy with a specific bias-column policy.
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Reports whether the generator has learned its output basis.
    pub fn is_fitted(&self) -> bool {
        self.n_features_in_.is_some() && self.n_output_features_.is_some() && self.powers_.is_some()
    }

    /// Builds the exponent basis used for polynomial expansion.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (_, cols) = ensure_2d_finite(x, "X")?;
        let powers = generate_powers(cols, self.degree, self.interaction_only, self.include_bias);
        let n_output_features = powers.len();
        let flat_powers = powers
            .into_iter()
            .flatten()
            .map(|value| value as f64)
            .collect::<Vec<_>>();

        self.n_features_in_ = Some(cols);
        self.n_output_features_ = Some(n_output_features);
        self.powers_ = Some(Array::from_shape_vec(
            &[n_output_features, cols],
            flat_powers,
        ));
        Ok(self)
    }

    /// Fits the generator and returns the expanded feature matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Expands each sample into the fitted polynomial basis.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("PolynomialFeatures"))?,
        )?;

        let powers = self
            .powers_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("PolynomialFeatures"))?;
        let n_output_features = self
            .n_output_features_
            .ok_or(PreprocessingError::NotFitted("PolynomialFeatures"))?;
        let mut data = Vec::with_capacity(rows * n_output_features);

        for row in 0..rows {
            let input_offset = row * cols;
            for feature in 0..n_output_features {
                let power_offset = feature * cols;
                let mut value = 1.0;
                for col in 0..cols {
                    let exponent = powers.data()[power_offset + col] as usize;
                    if exponent == 0 {
                        continue;
                    }

                    let base = x.data()[input_offset + col];
                    let mut factor = 1.0;
                    for _ in 0..exponent {
                        factor *= base;
                    }
                    value *= factor;
                }
                data.push(value);
            }
        }

        Ok(Array::from_shape_vec(&[rows, n_output_features], data))
    }
}
