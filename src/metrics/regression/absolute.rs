use crate::darray::Array;

use super::super::MetricsError;
use super::aggregation::{aggregate_output, weighted_median};
use super::context::RegressionContext;
use super::types::{RegressionMetricOptions, RegressionMetricOutput};
use wide::f64x4;

/// Returns the mean absolute error.
pub fn mean_absolute_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_absolute_error_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean absolute error with sklearn-style options.
pub fn mean_absolute_error_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    let values = per_output_absolute_values(&context);
    aggregate_output(values, &options.multioutput, None)
}

/// Returns the median absolute error.
pub fn median_absolute_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    median_absolute_error_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the median absolute error with sklearn-style options.
pub fn median_absolute_error_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    let values = per_output_median_absolute_values(&context);
    aggregate_output(values, &options.multioutput, None)
}

/// Returns the largest absolute residual.
pub fn max_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, None)?;
    if context.outputs != 1 {
        return Err(MetricsError::UnsupportedMultiOutput("max_error"));
    }

    Ok(context
        .y_true
        .data()
        .iter()
        .zip(context.y_pred.data())
        .map(|(true_value, pred_value)| (true_value - pred_value).abs())
        .fold(0.0, f64::max))
}

/// Returns the mean absolute percentage error.
pub fn mean_absolute_percentage_error(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_absolute_percentage_error_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean absolute percentage error with sklearn-style options.
pub fn mean_absolute_percentage_error_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    let values = per_output_percentage_values(&context);
    aggregate_output(values, &options.multioutput, None)
}

/// Computes one mean absolute error value per output column.
fn per_output_absolute_values(context: &RegressionContext<'_>) -> Vec<f64> {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();

    if context.outputs == 1 {
        let value = match context.sample_weights() {
            Some(weights) => weighted_single_output_absolute(y_true, y_pred, weights),
            None => simd_mean_absolute(y_true, y_pred),
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
            numerators[output] +=
                weight * (y_true[offset + output] - y_pred[offset + output]).abs();
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}

/// Computes one weighted median absolute error value per output column.
fn per_output_median_absolute_values(context: &RegressionContext<'_>) -> Vec<f64> {
    (0..context.outputs)
        .map(|output| {
            let mut errors = (0..context.samples)
                .map(|sample| {
                    (
                        (context.y_true_at(sample, output) - context.y_pred_at(sample, output))
                            .abs(),
                        context.sample_weight(sample),
                    )
                })
                .collect::<Vec<_>>();
            weighted_median(&mut errors)
        })
        .collect::<Vec<_>>()
}

/// Computes one mean absolute percentage error value per output column.
fn per_output_percentage_values(context: &RegressionContext<'_>) -> Vec<f64> {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();
    let mut numerators = vec![0.0; context.outputs];
    let mut denominators = vec![0.0; context.outputs];
    let weights = context.sample_weights();

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            let true_value = y_true[offset + output];
            let denominator = true_value.abs().max(f64::EPSILON);
            numerators[output] +=
                weight * (true_value - y_pred[offset + output]).abs() / denominator;
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}

fn simd_mean_absolute(y_true: &[f64], y_pred: &[f64]) -> f64 {
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
        numerator += (true_values - pred_values).abs();
        index += LANES;
    }

    let values: [f64; LANES] = numerator.into();
    let mut total = values.into_iter().sum::<f64>();
    while index < y_true.len() {
        total += (y_true[index] - y_pred[index]).abs();
        index += 1;
    }

    total / y_true.len() as f64
}

fn weighted_single_output_absolute(y_true: &[f64], y_pred: &[f64], weights: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for ((true_value, pred_value), weight) in y_true.iter().zip(y_pred).zip(weights) {
        numerator += weight * (true_value - pred_value).abs();
        denominator += weight;
    }

    numerator / denominator
}
