use crate::darray::Array;

use super::super::MetricsError;
use super::aggregation::{aggregate_output, weighted_median};
use super::context::RegressionContext;
use super::types::{RegressionMetricOptions, RegressionMetricOutput};

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

    Ok((0..context.samples)
        .map(|sample| (context.y_true_at(sample, 0) - context.y_pred_at(sample, 0)).abs())
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
    (0..context.outputs)
        .map(|output| {
            context.weighted_average(output, |sample, column| {
                (context.y_true_at(sample, column) - context.y_pred_at(sample, column)).abs()
            })
        })
        .collect::<Vec<_>>()
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
    (0..context.outputs)
        .map(|output| {
            context.weighted_average(output, |sample, column| {
                let true_value = context.y_true_at(sample, column);
                let denominator = true_value.abs().max(f64::EPSILON);
                (true_value - context.y_pred_at(sample, column)).abs() / denominator
            })
        })
        .collect::<Vec<_>>()
}
