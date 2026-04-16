use crate::darray::Array;

use super::super::PreprocessingError;
use super::super::common::{ensure_2d_finite, ensure_feature_count, unique_sorted};
use super::OrdinalHandleUnknown;

/// Encodes each categorical feature column as ordinal category indices.
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalEncoder {
    /// Optional manual category lists, one list per feature column.
    pub categories: Option<Vec<Vec<f64>>>,
    /// Controls how unseen categories are handled during transform.
    pub handle_unknown: OrdinalHandleUnknown,
    /// Encoded sentinel written for unseen categories when enabled.
    pub unknown_value: Option<f64>,
    /// Stores the learned category set for each input feature.
    pub categories_: Option<Vec<Vec<f64>>>,
    /// Stores the number of features observed during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for OrdinalEncoder {
    /// Returns the default ordinal encoding configuration.
    fn default() -> Self {
        Self {
            categories: None,
            handle_unknown: OrdinalHandleUnknown::Error,
            unknown_value: None,
            categories_: None,
            n_features_in_: None,
        }
    }
}

impl OrdinalEncoder {
    /// Creates an ordinal encoder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with manual per-feature category lists.
    pub fn categories(mut self, categories: Vec<Vec<f64>>) -> Self {
        self.categories = Some(categories);
        self
    }

    /// Returns a copy with a specific unknown-category policy.
    pub fn handle_unknown(mut self, handle_unknown: OrdinalHandleUnknown) -> Self {
        self.handle_unknown = handle_unknown;
        self
    }

    /// Returns a copy with a specific unknown-category sentinel.
    pub fn unknown_value(mut self, unknown_value: f64) -> Self {
        self.unknown_value = Some(unknown_value);
        self
    }

    /// Reports whether learned categories are available.
    pub fn is_fitted(&self) -> bool {
        self.categories_.is_some() && self.n_features_in_.is_some()
    }

    /// Learns sorted category vocabularies from each feature column.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        let categories = if let Some(categories) = &self.categories {
            if categories.len() != cols {
                return Err(PreprocessingError::InvalidCategories {
                    expected: cols,
                    got: categories.len(),
                });
            }

            let mut normalized = categories.clone();
            for feature_categories in &mut normalized {
                feature_categories.sort_by(f64::total_cmp);
                feature_categories.dedup_by(|left, right| left.total_cmp(right).is_eq());
            }
            normalized
        } else {
            (0..cols)
                .map(|col| unique_sorted(x, rows, cols, col))
                .collect::<Vec<_>>()
        };

        if self.handle_unknown == OrdinalHandleUnknown::UseEncodedValue {
            let unknown_value = self
                .unknown_value
                .ok_or(PreprocessingError::MissingUnknownValue)?;
            for feature_categories in &categories {
                if unknown_value >= 0.0
                    && unknown_value.fract() == 0.0
                    && (unknown_value as usize) < feature_categories.len()
                {
                    return Err(PreprocessingError::InvalidUnknownValue(unknown_value));
                }
            }
        }

        self.categories_ = Some(categories);
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Fits the encoder and returns the ordinal-encoded matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Maps each category to its ordinal index within that feature.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("OrdinalEncoder"))?,
        )?;

        let categories = self
            .categories_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("OrdinalEncoder"))?;
        let unknown_value = self.unknown_value;
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                let value = x.data()[offset + col];
                match categories[col].binary_search_by(|category| category.total_cmp(&value)) {
                    Ok(index) => data.push(index as f64),
                    Err(_) if self.handle_unknown == OrdinalHandleUnknown::UseEncodedValue => {
                        data.push(unknown_value.ok_or(PreprocessingError::MissingUnknownValue)?);
                    }
                    Err(_) => {
                        return Err(PreprocessingError::UnknownCategory {
                            feature_index: col,
                            value,
                        });
                    }
                }
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }

    /// Maps ordinal-encoded values back to their original categories.
    pub fn inverse_transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("OrdinalEncoder"))?,
        )?;

        let categories = self
            .categories_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("OrdinalEncoder"))?;
        let unknown_value = self.unknown_value;
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                let value = x.data()[offset + col];
                if self.handle_unknown == OrdinalHandleUnknown::UseEncodedValue
                    && Some(value) == unknown_value
                {
                    data.push(f64::NAN);
                    continue;
                }

                let index = value as usize;
                if value < 0.0 || value.fract() != 0.0 || index >= categories[col].len() {
                    return Err(PreprocessingError::InvalidEncodedLabel(value));
                }
                data.push(categories[col][index]);
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
