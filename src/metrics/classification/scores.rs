use crate::darray::Array;
use rayon::prelude::*;

use super::super::MetricsError;
use super::confusion::confusion_matrix_with_options;
use super::types::{
    AccuracyOptions, ClassificationAverage, ClassificationMetricOptions,
    ClassificationMetricOutput, ConfusionMatrixNormalize, ConfusionMatrixOptions,
};
use super::validation::{
    checked_zero_division, label_index, resolve_labels, validate_label_vectors,
    validate_sample_weight,
};

const PAR_THRESHOLD: usize = 16_384;

/// Returns the classification accuracy.
pub fn accuracy_score(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    accuracy_score_with_options(y_true, y_pred, AccuracyOptions::default())
}

/// Returns the classification accuracy with sklearn-style options.
pub fn accuracy_score_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: AccuracyOptions<'_>,
) -> Result<f64, MetricsError> {
    let samples = validate_label_vectors(y_true, y_pred)?;
    let sample_weight = validate_sample_weight(options.sample_weight, samples)?;
    let (correct, total) = if samples >= PAR_THRESHOLD {
        match sample_weight {
            Some(weights) => y_true
                .data()
                .par_iter()
                .zip(y_pred.data().par_iter())
                .zip(weights.par_iter())
                .map(|((true_value, pred_value), weight)| {
                    let correct = if true_value.total_cmp(pred_value).is_eq() {
                        *weight
                    } else {
                        0.0
                    };
                    (correct, *weight)
                })
                .reduce(
                    || (0.0, 0.0),
                    |left, right| (left.0 + right.0, left.1 + right.1),
                ),
            None => y_true
                .data()
                .par_iter()
                .zip(y_pred.data().par_iter())
                .map(|(true_value, pred_value)| {
                    if true_value.total_cmp(pred_value).is_eq() {
                        (1.0, 1.0)
                    } else {
                        (0.0, 1.0)
                    }
                })
                .reduce(
                    || (0.0, 0.0),
                    |left, right| (left.0 + right.0, left.1 + right.1),
                ),
        }
    } else {
        let mut correct = 0.0;
        let mut total = 0.0;
        for sample in 0..samples {
            let weight = sample_weight.map_or(1.0, |weights| weights[sample]);
            total += weight;
            if y_true.data()[sample]
                .total_cmp(&y_pred.data()[sample])
                .is_eq()
            {
                correct += weight;
            }
        }
        (correct, total)
    };

    if options.normalize {
        Ok(if total <= f64::EPSILON {
            0.0
        } else {
            correct / total
        })
    } else {
        Ok(correct)
    }
}

/// Returns the precision score.
pub fn precision_score(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    precision_score_with_options(y_true, y_pred, ClassificationMetricOptions::default())?
        .as_scalar()
        .ok_or(MetricsError::InvalidAverage("none"))
}

/// Returns the precision score with sklearn-style options.
pub fn precision_score_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: ClassificationMetricOptions<'_>,
) -> Result<ClassificationMetricOutput, MetricsError> {
    classification_metric(y_true, y_pred, options, MetricKind::Precision)
}

/// Returns the recall score.
pub fn recall_score(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    recall_score_with_options(y_true, y_pred, ClassificationMetricOptions::default())?
        .as_scalar()
        .ok_or(MetricsError::InvalidAverage("none"))
}

/// Returns the recall score with sklearn-style options.
pub fn recall_score_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: ClassificationMetricOptions<'_>,
) -> Result<ClassificationMetricOutput, MetricsError> {
    classification_metric(y_true, y_pred, options, MetricKind::Recall)
}

/// Returns the F1 score.
pub fn f1_score(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    f1_score_with_options(y_true, y_pred, ClassificationMetricOptions::default())?
        .as_scalar()
        .ok_or(MetricsError::InvalidAverage("none"))
}

/// Returns the F1 score with sklearn-style options.
pub fn f1_score_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: ClassificationMetricOptions<'_>,
) -> Result<ClassificationMetricOutput, MetricsError> {
    classification_metric(y_true, y_pred, options, MetricKind::F1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricKind {
    Precision,
    Recall,
    F1,
}

fn classification_metric(
    y_true: &Array,
    y_pred: &Array,
    options: ClassificationMetricOptions<'_>,
    kind: MetricKind,
) -> Result<ClassificationMetricOutput, MetricsError> {
    let zero_division = checked_zero_division(options.zero_division)?;
    let samples = validate_label_vectors(y_true, y_pred)?;
    validate_sample_weight(options.sample_weight, samples)?;
    let labels = resolve_labels(y_true, y_pred, options.labels)?;
    let confusion = confusion_matrix_with_options(
        y_true,
        y_pred,
        ConfusionMatrixOptions::default()
            .with_sample_weight_option(options.sample_weight)
            .with_labels_option(options.labels)
            .with_normalize(ConfusionMatrixNormalize::None),
    )?;
    let classes = labels.len();
    let mut precision = vec![0.0; classes];
    let mut recall = vec![0.0; classes];
    let mut f1 = vec![0.0; classes];
    let mut support = vec![0.0; classes];
    let mut predicted = vec![0.0; classes];
    let data = confusion.data();

    for class in 0..classes {
        support[class] = data[class * classes..(class + 1) * classes].iter().sum();
        predicted[class] = (0..classes).map(|row| data[row * classes + class]).sum();
        let tp = data[class * classes + class];
        precision[class] = safe_divide(tp, predicted[class], zero_division);
        recall[class] = safe_divide(tp, support[class], zero_division);
        let denom = precision[class] + recall[class];
        f1[class] = if denom <= f64::EPSILON {
            zero_division
        } else {
            2.0 * precision[class] * recall[class] / denom
        };
    }

    let values = match kind {
        MetricKind::Precision => precision,
        MetricKind::Recall => recall,
        MetricKind::F1 => f1,
    };
    let correct = (0..classes)
        .map(|class| data[class * classes + class])
        .sum::<f64>();

    aggregate_metric(
        &labels,
        &values,
        &support,
        correct,
        options.average,
        options.pos_label,
    )
}

fn aggregate_metric(
    labels: &[f64],
    values: &[f64],
    support: &[f64],
    correct: f64,
    average: ClassificationAverage,
    pos_label: f64,
) -> Result<ClassificationMetricOutput, MetricsError> {
    match average {
        ClassificationAverage::None => {
            Ok(ClassificationMetricOutput::PerClass(Array::array(values)))
        }
        ClassificationAverage::Binary => {
            if labels.len() > 2 {
                return Err(MetricsError::InvalidAverage("binary"));
            }
            let index =
                label_index(labels, pos_label).ok_or(MetricsError::UnknownLabel(pos_label))?;
            Ok(ClassificationMetricOutput::Scalar(values[index]))
        }
        ClassificationAverage::Macro => Ok(ClassificationMetricOutput::Scalar(mean(values))),
        ClassificationAverage::Weighted => Ok(ClassificationMetricOutput::Scalar(weighted_mean(
            values, support,
        ))),
        ClassificationAverage::Micro => {
            let total_support = support.iter().sum::<f64>();
            let micro = if total_support <= f64::EPSILON {
                0.0
            } else {
                correct / total_support
            };
            Ok(ClassificationMetricOutput::Scalar(micro))
        }
    }
}

fn safe_divide(numerator: f64, denominator: f64, zero_division: f64) -> f64 {
    if denominator <= f64::EPSILON {
        zero_division
    } else {
        numerator / denominator
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let total_weight = weights.iter().sum::<f64>();
    if total_weight <= f64::EPSILON {
        return mean(values);
    }
    values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / total_weight
}

trait ConfusionOptionExt<'a> {
    fn with_sample_weight_option(self, sample_weight: Option<&'a Array>) -> Self;
    fn with_labels_option(self, labels: Option<&'a Array>) -> Self;
}

impl<'a> ConfusionOptionExt<'a> for ConfusionMatrixOptions<'a> {
    fn with_sample_weight_option(mut self, sample_weight: Option<&'a Array>) -> Self {
        self.sample_weight = sample_weight;
        self
    }

    fn with_labels_option(mut self, labels: Option<&'a Array>) -> Self {
        self.labels = labels;
        self
    }
}
