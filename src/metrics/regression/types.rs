use crate::darray::Array;

use super::super::MetricsError;

/// Controls how multi-output regression metrics are aggregated.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiOutput {
    /// Returns one metric value per output column.
    RawValues,
    /// Returns the unweighted mean across output columns.
    UniformAverage,
    /// Returns a weighted mean using per-output target variance.
    VarianceWeighted,
    /// Returns a weighted mean using explicit output weights.
    Custom(Array),
}

/// Optional arguments for regression metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionMetricOptions<'a> {
    pub sample_weight: Option<&'a Array>,
    pub multioutput: MultiOutput,
}

impl<'a> Default for RegressionMetricOptions<'a> {
    /// Builds options matching sklearn's default regression metric behavior.
    fn default() -> Self {
        Self {
            sample_weight: None,
            multioutput: MultiOutput::UniformAverage,
        }
    }
}

impl<'a> RegressionMetricOptions<'a> {
    /// Creates default options matching sklearn's default aggregation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets per-sample weights.
    pub fn with_sample_weight(mut self, sample_weight: &'a Array) -> Self {
        self.sample_weight = Some(sample_weight);
        self
    }

    /// Sets the multi-output aggregation policy.
    pub fn with_multioutput(mut self, multioutput: MultiOutput) -> Self {
        self.multioutput = multioutput;
        self
    }
}

/// Returns either a scalar metric or one value per output.
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionMetricOutput {
    Scalar(f64),
    RawValues(Array),
}

impl RegressionMetricOutput {
    /// Returns the scalar value when the metric used aggregated output.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Self::Scalar(value) => Some(*value),
            Self::RawValues(_) => None,
        }
    }

    /// Returns the per-output values when the metric used raw output.
    pub fn as_raw_values(&self) -> Option<&Array> {
        match self {
            Self::Scalar(_) => None,
            Self::RawValues(values) => Some(values),
        }
    }

    /// Extracts a scalar metric value or reports that raw values were requested.
    pub(crate) fn into_scalar(self) -> Result<f64, MetricsError> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::RawValues(_) => Err(MetricsError::UnsupportedMultiOutput(
                "this metric result was requested as raw values",
            )),
        }
    }
}
