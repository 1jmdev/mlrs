use crate::darray::Array;

use super::super::MetricsError;
use super::types::{MultiOutput, RegressionMetricOutput};
use super::validation::validate_finite_array;

/// Aggregates per-output values according to the requested policy.
pub(crate) fn aggregate_output(
    values: Vec<f64>,
    multioutput: &MultiOutput,
    variance_weights: Option<&[f64]>,
) -> Result<RegressionMetricOutput, MetricsError> {
    match multioutput {
        MultiOutput::RawValues => Ok(RegressionMetricOutput::RawValues(Array::array(&values))),
        MultiOutput::UniformAverage => Ok(RegressionMetricOutput::Scalar(mean_slice(&values))),
        MultiOutput::VarianceWeighted => {
            let weights = variance_weights.ok_or(MetricsError::UnsupportedMultiOutput(
                "this metric does not support variance-weighted aggregation",
            ))?;
            Ok(RegressionMetricOutput::Scalar(weighted_mean_slice(
                &values, weights,
            )))
        }
        MultiOutput::Custom(weights) => {
            if !weights.is_vector() {
                return Err(MetricsError::InvalidInputShape(weights.shape().to_vec()));
            }
            if weights.len() != values.len() {
                return Err(MetricsError::InvalidMultiOutputWeights {
                    expected: values.len(),
                    got: weights.len(),
                });
            }
            validate_finite_array(weights, "multioutput")?;
            Ok(RegressionMetricOutput::Scalar(weighted_mean_slice(
                &values,
                weights.data(),
            )))
        }
    }
}

/// Returns the arithmetic mean of a slice.
pub(crate) fn mean_slice(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Returns the weighted mean of a slice.
pub(crate) fn weighted_mean_slice(values: &[f64], weights: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (value, weight) in values.iter().zip(weights) {
        numerator += value * weight;
        denominator += weight;
    }

    if denominator <= f64::EPSILON {
        mean_slice(values)
    } else {
        numerator / denominator
    }
}

/// Returns the weighted median of value-weight pairs.
pub(crate) fn weighted_median(values: &mut [(f64, f64)]) -> f64 {
    values.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight = values.iter().map(|(_, weight)| *weight).sum::<f64>();
    let threshold = total_weight / 2.0;
    let mut cumulative = 0.0;

    for (value, weight) in values.iter() {
        cumulative += *weight;
        if cumulative >= threshold {
            return *value;
        }
    }

    values.last().map(|(value, _)| *value).unwrap_or(f64::NAN)
}

/// Converts explained variance sums into a final score.
pub(crate) fn explained_variance_from_sums(numerator: f64, denominator: f64) -> f64 {
    if denominator <= f64::EPSILON {
        if numerator <= f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - numerator / denominator
    }
}

/// Converts residual and total sums into an `R^2` score.
pub(crate) fn r2_from_sums(numerator: f64, denominator: f64) -> f64 {
    if denominator <= f64::EPSILON {
        if numerator <= f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - numerator / denominator
    }
}

/// Returns the per-sample Tweedie unit deviance.
pub(crate) fn tweedie_unit_deviance(y_true: f64, y_pred: f64, power: f64) -> f64 {
    if power == 0.0 {
        let delta = y_true - y_pred;
        return delta * delta;
    }

    if power == 1.0 {
        if y_true == 0.0 {
            return 2.0 * y_pred;
        }
        return 2.0 * (y_true * (y_true / y_pred).ln() - y_true + y_pred);
    }

    if power == 2.0 {
        return 2.0 * ((y_true / y_pred) - (y_true / y_pred).ln() - 1.0);
    }

    let one_minus_power = 1.0 - power;
    let two_minus_power = 2.0 - power;
    2.0 * (y_true.powf(two_minus_power) / (one_minus_power * two_minus_power)
        - y_true * y_pred.powf(one_minus_power) / one_minus_power
        + y_pred.powf(two_minus_power) / two_minus_power)
}
