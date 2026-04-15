use crate::darray::Array;

use super::super::MetricsError;
use super::aggregation::{aggregate_output, explained_variance_from_sums, r2_from_sums};
use super::context::RegressionContext;
use super::types::{RegressionMetricOptions, RegressionMetricOutput};

/// Returns the explained variance score.
pub fn explained_variance_score(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    explained_variance_score_with_options(y_true, y_pred, RegressionMetricOptions::default())?
        .into_scalar()
}

/// Returns the explained variance score with sklearn-style options.
pub fn explained_variance_score_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    let (values, denominators) = explained_variance_components(&context);
    aggregate_output(values, &options.multioutput, Some(&denominators))
}

/// Returns the coefficient of determination.
pub fn r2_score(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    r2_score_with_options(y_true, y_pred, RegressionMetricOptions::default())?.into_scalar()
}

/// Returns the coefficient of determination with sklearn-style options.
pub fn r2_score_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: RegressionMetricOptions<'_>,
) -> Result<RegressionMetricOutput, MetricsError> {
    let context = RegressionContext::new(y_true, y_pred, options.sample_weight)?;
    let (values, denominators) = r2_components(&context);
    aggregate_output(values, &options.multioutput, Some(&denominators))
}

/// Computes per-output explained variance values and variance weights.
fn explained_variance_components(context: &RegressionContext<'_>) -> (Vec<f64>, Vec<f64>) {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();
    let weights = context.sample_weights();
    let mut target_sums = vec![0.0; context.outputs];
    let mut residual_sums = vec![0.0; context.outputs];
    let mut denominators = Vec::with_capacity(context.outputs);
    let mut total_weight = 0.0;

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        total_weight += weight;
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            let true_value = y_true[offset + output];
            target_sums[output] += weight * true_value;
            residual_sums[output] += weight * (true_value - y_pred[offset + output]);
        }
    }

    let target_means = target_sums
        .iter()
        .map(|sum| sum / total_weight)
        .collect::<Vec<_>>();
    let residual_means = residual_sums
        .iter()
        .map(|sum| sum / total_weight)
        .collect::<Vec<_>>();

    let mut numerators = vec![0.0; context.outputs];
    let mut denominator_sums = vec![0.0; context.outputs];

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            let true_value = y_true[offset + output];
            let residual = true_value - y_pred[offset + output] - residual_means[output];
            let centered = true_value - target_means[output];
            numerators[output] += weight * residual * residual;
            denominator_sums[output] += weight * centered * centered;
        }
    }

    let values = numerators
        .into_iter()
        .zip(denominator_sums.iter().copied())
        .map(|(numerator, denominator)| {
            explained_variance_from_sums(numerator / total_weight, denominator / total_weight)
        })
        .collect::<Vec<_>>();

    denominators.extend(denominator_sums.into_iter().map(|sum| sum / total_weight));

    (values, denominators)
}

/// Computes per-output `R^2` values and variance weights.
fn r2_components(context: &RegressionContext<'_>) -> (Vec<f64>, Vec<f64>) {
    let y_true = context.y_true.data();
    let y_pred = context.y_pred.data();
    let weights = context.sample_weights();
    let mut target_sums = vec![0.0; context.outputs];
    let mut total_weight = 0.0;

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        total_weight += weight;
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            target_sums[output] += weight * y_true[offset + output];
        }
    }

    let means = target_sums
        .iter()
        .map(|sum| sum / total_weight)
        .collect::<Vec<_>>();
    let mut numerators = vec![0.0; context.outputs];
    let mut denominator_sums = vec![0.0; context.outputs];

    for sample in 0..context.samples {
        let weight = weights.map_or(1.0, |values| values[sample]);
        let offset = sample * context.outputs;
        for output in 0..context.outputs {
            let delta = y_true[offset + output] - y_pred[offset + output];
            let centered = y_true[offset + output] - means[output];
            numerators[output] += weight * delta * delta;
            denominator_sums[output] += weight * centered * centered;
        }
    }

    let values = numerators
        .into_iter()
        .zip(denominator_sums.iter().copied())
        .map(|(numerator, denominator)| {
            r2_from_sums(numerator / total_weight, denominator / total_weight)
        })
        .collect::<Vec<_>>();
    let denominators = denominator_sums
        .into_iter()
        .map(|sum| sum / total_weight)
        .collect::<Vec<_>>();

    (values, denominators)
}
