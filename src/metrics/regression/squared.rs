use crate::darray::Array;

use super::super::MetricsError;
use super::aggregation::aggregate_output;
use super::context::RegressionContext;
use super::types::{RegressionMetricOptions, RegressionMetricOutput};
use super::validation::validate_non_negative_shifted_inputs;
use wide::f64x4;

/// Returns the mean squared error.
pub fn mean_squared_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_squared_error_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean squared error with sklearn-style options.
pub fn mean_squared_error_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    let values = per_output_squared_values(&context);
    aggregate_output(values, &options.multioutput, None)
}

/// Returns the root mean squared error.
pub fn root_mean_squared_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    root_mean_squared_error_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the root mean squared error with sklearn-style options.
pub fn root_mean_squared_error_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let mse = mean_squared_error_with_options(y_true, y_pred, options)?;
    Ok(sqrt_metric_output(mse))
}

/// Returns the mean squared logarithmic error.
pub fn mean_squared_log_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_squared_log_error_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean squared logarithmic error with sklearn-style options.
pub fn mean_squared_log_error_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    validate_non_negative_shifted_inputs(&context, "mean_squared_log_error")?;
    let values = per_output_squared_log_values(&context);
    aggregate_output(values, &options.multioutput, None)
}

/// Returns the root mean squared logarithmic error.
pub fn root_mean_squared_log_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    root_mean_squared_log_error_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the root mean squared logarithmic error with sklearn-style options.
pub fn root_mean_squared_log_error_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let msle = mean_squared_log_error_with_options(y_true, y_pred, options)?;
    Ok(sqrt_metric_output(msle))
}

/// Computes one mean squared error value per output column.
fn per_output_squared_values(context: &RegressionContext<'_>) -> Vec<f64> {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();

    if context.outputs == 1 {
        let value = match context.sample_weights() {
            Some(weights) => weighted_single_output_squared(y_true, y_pred, weights),
            None => simd_mean_squared(y_true, y_pred),
        };
        return vec![value];
    }

    let mut numerators = vec![0.0; context.outputs];
    let mut denominators = vec![0.0; context.outputs];
    let weights = context.sample_weights();

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            let delta = y_true[offset + output] - y_pred[offset + output];
            numerators[output] += weight * delta * delta;
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}

/// Computes one mean squared logarithmic error value per output column.
fn per_output_squared_log_values(context: &RegressionContext<'_>) -> Vec<f64> {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();
    let mut numerators = vec![0.0; context.outputs];
    let mut denominators = vec![0.0; context.outputs];
    let weights = context.sample_weights();

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            let delta = y_true[offset + output].ln_1p() - y_pred[offset + output].ln_1p();
            numerators[output] += weight * delta * delta;
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}

fn simd_mean_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    const LANES: usize = 4;
    let mut numerator = f64x4::from([0.0; LANES]);
    let mut index = 0;

    while index + LANES <= y_true.len() {
        let true_values = f64x4::from([
            y_true[index],
            y_true[index + 1],
            y_true[index + 2],
            y_true[index + 3],
        ]);
        let pred_values = f64x4::from([
            y_pred[index],
            y_pred[index + 1],
            y_pred[index + 2],
            y_pred[index + 3],
        ]);
        let delta = true_values - pred_values;
        numerator += delta * delta;
        index += LANES;
    }

    let values: [f64; LANES] = numerator.into();
    let mut total = values.into_iter().sum::<f64>();
    while index < y_true.len() {
        let delta = y_true[index] - y_pred[index];
        total += delta * delta;
        index += 1;
    }

    total / y_true.len() as f64
}

fn weighted_single_output_squared(y_true: &[f64], y_pred: &[f64], weights: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for ((true_value, pred_value), weight) in y_true.iter().zip(y_pred).zip(weights) {
        let delta = true_value - pred_value;
        numerator += weight * delta * delta;
        denominator += weight;
    }

    numerator / denominator
}

/// Applies a square root to scalar or raw multi-output metric values.
fn sqrt_metric_output(output: RegressionMetricOutput) -> RegressionMetricOutput {
    match output {
        RegressionMetricOutput::Scalar(value) => RegressionMetricOutput::Scalar(value.sqrt()),
        RegressionMetricOutput::RawValues(values) => {
            RegressionMetricOutput::RawValues(values.sqrt())
        }
    }
}
