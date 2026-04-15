use crate::darray::Array;

use super::super::MetricsError;
use super::aggregation::aggregate_output;
use super::context::RegressionContext;
use super::types::{RegressionMetricOptions, RegressionMetricOutput};
use super::validation::validate_non_negative_shifted_inputs;

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
    (0..context.outputs)
        .map(|output| {
            context.weighted_average(output, |sample, column| {
                let delta = context.y_true_at(sample, column) - context.y_pred_at(sample, column);
                delta * delta
            })
        })
        .collect::<Vec<_>>()
}

/// Computes one mean squared logarithmic error value per output column.
fn per_output_squared_log_values(context: &RegressionContext<'_>) -> Vec<f64> {
    (0..context.outputs)
        .map(|output| {
            context.weighted_average(output, |sample, column| {
                let true_value = context.y_true_at(sample, column).ln_1p();
                let pred_value = context.y_pred_at(sample, column).ln_1p();
                let delta = true_value - pred_value;
                delta * delta
            })
        })
        .collect::<Vec<_>>()
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
