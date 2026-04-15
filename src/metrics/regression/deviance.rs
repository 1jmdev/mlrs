use crate::darray::Array;

use super::super::MetricsError;
use super::aggregation::{aggregate_output, tweedie_unit_deviance};
use super::context::RegressionContext;
use super::types::{RegressionMetricOptions, RegressionMetricOutput};
use super::validation::{validate_gamma_inputs, validate_poisson_inputs, validate_tweedie_inputs};

/// Returns the mean pinball loss.
pub fn mean_pinball_loss(y_true: &Array, y_pred: &Array, alpha: f64) -> Result<f64, MetricsError> {
    mean_pinball_loss_with_options(y_true, y_pred, alpha, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean pinball loss with sklearn-style options.
pub fn mean_pinball_loss_with_options(
    y_true: &Array,
    y_pred: &Array,
    alpha: f64,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(MetricsError::InvalidAlpha(alpha));
    }

    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    let values = per_output_pinball_values(&context, alpha);
    aggregate_output(values, &options.multioutput, None)
}

/// Returns the mean Poisson deviance.
pub fn mean_poisson_deviance(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_poisson_deviance_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean Poisson deviance with sklearn-style options.
pub fn mean_poisson_deviance_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    validate_poisson_inputs(&context)?;
    let values = per_output_poisson_values(&context);
    aggregate_output(values, &options.multioutput, None)
}

/// Returns the mean Gamma deviance.
pub fn mean_gamma_deviance(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_gamma_deviance_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean Gamma deviance with sklearn-style options.
pub fn mean_gamma_deviance_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    validate_gamma_inputs(&context)?;
    let values = per_output_gamma_values(&context);
    aggregate_output(values, &options.multioutput, None)
}

/// Returns the mean Tweedie deviance.
pub fn mean_tweedie_deviance(
    y_true: &Array,
    y_pred: &Array,
    power: f64,
) -> Result<f64, MetricsError> {
    mean_tweedie_deviance_with_options(y_true, y_pred, power, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the mean Tweedie deviance with sklearn-style options.
pub fn mean_tweedie_deviance_with_options(
    y_true: &Array,
    y_pred: &Array,
    power: f64,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    validate_tweedie_inputs(&context, power)?;
    let values = per_output_tweedie_values(&context, power);
    aggregate_output(values, &options.multioutput, None)
}

/// Computes one mean pinball loss value per output column.
fn per_output_pinball_values(context: &RegressionContext<'_>, alpha: f64) -> Vec<f64> {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();
    let mut numerators = vec![0.0; context.outputs];
    let mut denominators = vec![0.0; context.outputs];
    let weights = context.sample_weights();

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            let delta = y_true[offset + output] - y_pred[offset + output];
            let value = if delta >= 0.0 {
                alpha * delta
            } else {
                (alpha - 1.0) * delta
            };
            numerators[output] += weight * value;
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}

/// Computes one mean Poisson deviance value per output column.
fn per_output_poisson_values(context: &RegressionContext<'_>) -> Vec<f64> {
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
            let pred_value = y_pred[offset + output];
            let value = if true_value == 0.0 {
                2.0 * pred_value
            } else {
                2.0 * (true_value * (true_value / pred_value).ln() - true_value + pred_value)
            };
            numerators[output] += weight * value;
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}

/// Computes one mean Gamma deviance value per output column.
fn per_output_gamma_values(context: &RegressionContext<'_>) -> Vec<f64> {
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
            let pred_value = y_pred[offset + output];
            let value =
                2.0 * (((true_value - pred_value) / pred_value) - (true_value / pred_value).ln());
            numerators[output] += weight * value;
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}

/// Computes one mean Tweedie deviance value per output column.
fn per_output_tweedie_values(context: &RegressionContext<'_>, power: f64) -> Vec<f64> {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();
    let mut numerators = vec![0.0; context.outputs];
    let mut denominators = vec![0.0; context.outputs];
    let weights = context.sample_weights();

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            numerators[output] += weight
                * tweedie_unit_deviance(y_true[offset + output], y_pred[offset + output], power);
            denominators[output] += weight;
        }
    }

    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator)| numerator / denominator)
        .collect()
}
