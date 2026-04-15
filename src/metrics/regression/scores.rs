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
    let mut values = Vec::with_capacity(context.outputs);
    let mut denominators = Vec::with_capacity(context.outputs);

    for output in 0..context.outputs {
        let target_mean =
            context.weighted_average(output, |sample, column| context.y_true_at(sample, column));
        let residual_mean = context.weighted_average(output, |sample, column| {
            context.y_true_at(sample, column) - context.y_pred_at(sample, column)
        });
        let numerator = context.weighted_average(output, |sample, column| {
            let residual = context.y_true_at(sample, column)
                - context.y_pred_at(sample, column)
                - residual_mean;
            residual * residual
        });
        let denominator = context.weighted_average(output, |sample, column| {
            let centered = context.y_true_at(sample, column) - target_mean;
            centered * centered
        });

        denominators.push(denominator);
        values.push(explained_variance_from_sums(numerator, denominator));
    }

    (values, denominators)
}

/// Computes per-output `R^2` values and variance weights.
fn r2_components(context: &RegressionContext<'_>) -> (Vec<f64>, Vec<f64>) {
    let mut values = Vec::with_capacity(context.outputs);
    let mut denominators = Vec::with_capacity(context.outputs);

    for output in 0..context.outputs {
        let mean =
            context.weighted_average(output, |sample, column| context.y_true_at(sample, column));
        let numerator = context.weighted_average(output, |sample, column| {
            let delta = context.y_true_at(sample, column) - context.y_pred_at(sample, column);
            delta * delta
        });
        let denominator = context.weighted_average(output, |sample, column| {
            let centered = context.y_true_at(sample, column) - mean;
            centered * centered
        });

        denominators.push(denominator);
        values.push(r2_from_sums(numerator, denominator));
    }

    (values, denominators)
}
