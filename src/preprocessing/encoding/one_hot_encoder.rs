use crate::darray::Array;

use super::super::PreprocessingError;
use super::super::common::{ensure_2d_finite, ensure_feature_count, unique_sorted};
use super::HandleUnknown;

/// Encodes categorical feature columns as a dense one-hot matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OneHotEncoder {
    /// Controls how unseen categories are handled during transform.
    pub handle_unknown: HandleUnknown,
    /// Optional manual category lists, one list per feature column.
    pub categories: Option<Vec<Vec<f64>>>,
    /// Stores the learned category set for each input feature.
    pub categories_: Option<Vec<Vec<f64>>>,
    /// Stores the width of each encoded feature block.
    pub feature_sizes_: Option<Vec<usize>>,
    /// Stores the number of features observed during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for OneHotEncoder {
    /// Returns the default one-hot encoding configuration.
    fn default() -> Self {
        Self {
            handle_unknown: HandleUnknown::Error,
            categories: None,
            categories_: None,
            feature_sizes_: None,
            n_features_in_: None,
        }
    }
}

impl OneHotEncoder {
    /// Creates a one-hot encoder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific unknown-category policy.
    pub fn handle_unknown(mut self, handle_unknown: HandleUnknown) -> Self {
        self.handle_unknown = handle_unknown;
        self
    }

    /// Returns a copy with manual per-feature category lists.
    pub fn categories(mut self, categories: Vec<Vec<f64>>) -> Self {
        self.categories = Some(categories);
        self
    }

    /// Reports whether learned categories are available.
    pub fn is_fitted(&self) -> bool {
        self.categories_.is_some() && self.feature_sizes_.is_some() && self.n_features_in_.is_some()
    }

    /// Learns one-hot category vocabularies from each feature column.
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

        let feature_sizes = categories.iter().map(Vec::len).collect::<Vec<_>>();
        self.categories_ = Some(categories);
        self.feature_sizes_ = Some(feature_sizes);
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Fits the encoder and returns the dense one-hot matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Expands categorical features into dense one-hot columns.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_finite(x, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("OneHotEncoder"))?,
        )?;

        let categories = self
            .categories_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("OneHotEncoder"))?;
        let feature_sizes = self
            .feature_sizes_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("OneHotEncoder"))?;
        let encoded_cols = feature_sizes.iter().sum();
        let mut data = vec![0.0; rows * encoded_cols];

        for row in 0..rows {
            let mut output_col = 0;
            for col in 0..cols {
                let value = x.data()[row * cols + col];
                match categories[col].binary_search_by(|category| category.total_cmp(&value)) {
                    Ok(position) => {
                        data[row * encoded_cols + output_col + position] = 1.0;
                    }
                    Err(_) if self.handle_unknown == HandleUnknown::Error => {
                        return Err(PreprocessingError::UnknownCategory {
                            feature_index: col,
                            value,
                        });
                    }
                    Err(_) => {}
                }
                output_col += feature_sizes[col];
            }
        }

        Ok(Array::from_shape_vec(&[rows, encoded_cols], data))
    }

    /// Decodes a dense one-hot matrix back to the original category matrix.
    pub fn inverse_transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, encoded_cols) = ensure_2d_finite(x, "X")?;
        let categories = self
            .categories_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("OneHotEncoder"))?;
        let feature_sizes = self
            .feature_sizes_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("OneHotEncoder"))?;
        let cols = self
            .n_features_in_
            .ok_or(PreprocessingError::NotFitted("OneHotEncoder"))?;
        let expected_cols = feature_sizes.iter().sum::<usize>();
        ensure_feature_count(encoded_cols, expected_cols)?;

        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let mut encoded_offset = 0;
            for col in 0..cols {
                let width = feature_sizes[col];
                let segment = &x.data()[row * encoded_cols + encoded_offset
                    ..row * encoded_cols + encoded_offset + width];
                let mut active = None;
                let mut multiple = false;
                for (index, value) in segment.iter().enumerate() {
                    if *value == 0.0 {
                        continue;
                    }
                    if active.is_some() {
                        multiple = true;
                        break;
                    }
                    active = Some(index);
                }

                let value = match (active, multiple) {
                    (Some(index), false) => categories[col][index],
                    (None, false) if self.handle_unknown == HandleUnknown::Ignore => f64::NAN,
                    (None, false) => {
                        return Err(PreprocessingError::InvalidEncodedRow {
                            sample_index: row,
                            feature_index: col,
                            details: "no active category in encoded segment",
                        });
                    }
                    _ => {
                        return Err(PreprocessingError::InvalidEncodedRow {
                            sample_index: row,
                            feature_index: col,
                            details: "multiple active categories in encoded segment",
                        });
                    }
                };

                data.push(value);
                encoded_offset += width;
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}
