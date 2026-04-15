use crate::darray::Array;

use super::super::common::{ensure_1d_finite, unique_sorted_1d};
use super::super::PreprocessingError;

/// Encodes a one-dimensional target vector into contiguous class indices.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelEncoder {
    /// Stores the fitted class labels in sorted order.
    pub classes_: Option<Array>,
}

impl Default for LabelEncoder {
    /// Returns the default label encoding configuration.
    fn default() -> Self {
        Self { classes_: None }
    }
}

impl LabelEncoder {
    /// Creates a label encoder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reports whether fitted classes are available.
    pub fn is_fitted(&self) -> bool {
        self.classes_.is_some()
    }

    /// Learns sorted unique classes from a label vector.
    pub fn fit(&mut self, y: &Array) -> Result<&mut Self, PreprocessingError> {
        ensure_1d_finite(y, "y")?;
        let classes = unique_sorted_1d(y);
        self.classes_ = Some(Array::from_shape_vec(&[classes.len()], classes));
        Ok(self)
    }

    /// Fits the encoder and returns encoded class indices.
    pub fn fit_transform(&mut self, y: &Array) -> Result<Array, PreprocessingError> {
        self.fit(y)?;
        self.transform(y)
    }

    /// Maps labels to contiguous encoded class indices.
    pub fn transform(&self, y: &Array) -> Result<Array, PreprocessingError> {
        let n_samples = ensure_1d_finite(y, "y")?;
        let classes = self
            .classes_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("LabelEncoder"))?
            .data();
        let mut data = Vec::with_capacity(n_samples);

        for &value in y.data() {
            let index = classes
                .binary_search_by(|class| class.total_cmp(&value))
                .map_err(|_| PreprocessingError::UnknownLabel(value))?;
            data.push(index as f64);
        }

        Ok(Array::from_shape_vec(&[n_samples], data))
    }

    /// Maps encoded class indices back to their original labels.
    pub fn inverse_transform(&self, y: &Array) -> Result<Array, PreprocessingError> {
        let n_samples = ensure_1d_finite(y, "y")?;
        let classes = self
            .classes_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("LabelEncoder"))?
            .data();
        let mut data = Vec::with_capacity(n_samples);

        for &value in y.data() {
            let index = value as usize;
            if value < 0.0 || value.fract() != 0.0 || index >= classes.len() {
                return Err(PreprocessingError::InvalidEncodedLabel(value));
            }
            data.push(classes[index]);
        }

        Ok(Array::from_shape_vec(&[n_samples], data))
    }
}
