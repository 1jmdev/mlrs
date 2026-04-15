use crate::darray::Array;

use super::super::MetricsError;
use super::validation::{validate_finite_array, validate_regression_shapes};

/// Carries validated regression inputs and derived shape information.
pub(crate) struct RegressionContext<'a> {
    pub(crate) y_true: &'a Array,
    pub(crate) y_pred: &'a Array,
    sample_weight: Option<&'a Array>,
    pub(crate) samples: usize,
    pub(crate) outputs: usize,
}

impl<'a> RegressionContext<'a> {
    /// Validates the metric inputs and prepares shared indexing state.
    pub(crate) fn new(
        y_true: &'a Array,
        y_pred: &'a Array,
        sample_weight: Option<&'a Array>,
    ) -> Result<Self, MetricsError> {
        validate_regression_shapes(y_true, y_pred)?;
        validate_finite_array(y_true, "y_true")?;
        validate_finite_array(y_pred, "y_pred")?;

        let samples = y_true.shape()[0];
        let outputs = if y_true.ndim() == 1 {
            1
        } else {
            y_true.shape()[1]
        };

        if let Some(weights) = sample_weight {
            if !weights.is_vector() {
                return Err(MetricsError::InvalidSampleWeightShape(
                    weights.shape().to_vec(),
                ));
            }
            if weights.len() != samples {
                return Err(MetricsError::SampleWeightLengthMismatch {
                    expected: samples,
                    got: weights.len(),
                });
            }
            validate_finite_array(weights, "sample_weight")?;
        }

        Ok(Self {
            y_true,
            y_pred,
            sample_weight,
            samples,
            outputs,
        })
    }

    /// Returns a target value for one sample-output coordinate.
    pub(crate) fn y_true_at(&self, sample: usize, output: usize) -> f64 {
        if self.outputs == 1 && self.y_true.ndim() == 1 {
            self.y_true.data()[sample]
        } else {
            self.y_true.data()[sample * self.outputs + output]
        }
    }

    /// Returns a prediction value for one sample-output coordinate.
    pub(crate) fn y_pred_at(&self, sample: usize, output: usize) -> f64 {
        if self.outputs == 1 && self.y_pred.ndim() == 1 {
            self.y_pred.data()[sample]
        } else {
            self.y_pred.data()[sample * self.outputs + output]
        }
    }

    /// Returns the sample weight for one row, defaulting to `1.0`.
    pub(crate) fn sample_weight(&self, sample: usize) -> f64 {
        self.sample_weight
            .map_or(1.0, |weights| weights.data()[sample])
    }

    /// Returns the validated sample-weight slice when present.
    pub(crate) fn sample_weights(&self) -> Option<&[f64]> {
        self.sample_weight.map(Array::data)
    }
}
