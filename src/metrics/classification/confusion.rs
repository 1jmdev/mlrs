use crate::darray::Array;
use rayon::prelude::*;

use super::super::MetricsError;
use super::types::{ConfusionMatrixNormalize, ConfusionMatrixOptions};
use super::validation::{
    build_label_lookup, label_lookup_index, resolve_labels, validate_label_vectors,
    validate_sample_weight,
};

const PAR_THRESHOLD: usize = 16_384;

/// Returns the confusion matrix for one-dimensional class labels.
pub fn confusion_matrix(y_true: &Array, y_pred: &Array) -> Result<Array, MetricsError> {
    confusion_matrix_with_options(y_true, y_pred, ConfusionMatrixOptions::default())
}

/// Returns the confusion matrix with sklearn-style options.
pub fn confusion_matrix_with_options(
    y_true: &Array,
    y_pred: &Array,
    options: ConfusionMatrixOptions<'_>,
) -> Result<Array, MetricsError> {
    let samples = validate_label_vectors(y_true, y_pred)?;
    let sample_weight = validate_sample_weight(options.sample_weight, samples)?;
    let labels = resolve_labels(y_true, y_pred, options.labels)?;
    let label_lookup = build_label_lookup(&labels);
    let classes = labels.len();
    let mut data = if samples >= PAR_THRESHOLD {
        (0..samples)
            .into_par_iter()
            .fold(
                || vec![0.0; classes * classes],
                |mut local, sample| {
                    update_confusion_counts(
                        &mut local,
                        &label_lookup,
                        y_true.data()[sample],
                        y_pred.data()[sample],
                        sample_weight.map_or(1.0, |weights| weights[sample]),
                        classes,
                    );
                    local
                },
            )
            .reduce(
                || vec![0.0; classes * classes],
                |mut left, right| {
                    for (left_value, right_value) in left.iter_mut().zip(right) {
                        *left_value += right_value;
                    }
                    left
                },
            )
    } else {
        let mut local = vec![0.0; classes * classes];
        for sample in 0..samples {
            update_confusion_counts(
                &mut local,
                &label_lookup,
                y_true.data()[sample],
                y_pred.data()[sample],
                sample_weight.map_or(1.0, |weights| weights[sample]),
                classes,
            );
        }
        local
    };

    normalize_confusion_matrix(&mut data, classes, options.normalize);
    Ok(Array::from_shape_vec(&[classes, classes], data))
}

fn update_confusion_counts(
    data: &mut [f64],
    label_lookup: &[(f64, usize)],
    y_true: f64,
    y_pred: f64,
    weight: f64,
    classes: usize,
) {
    let Some(true_index) = label_lookup_index(label_lookup, y_true) else {
        return;
    };
    let Some(pred_index) = label_lookup_index(label_lookup, y_pred) else {
        return;
    };
    data[true_index * classes + pred_index] += weight;
}

fn normalize_confusion_matrix(
    data: &mut [f64],
    classes: usize,
    normalize: ConfusionMatrixNormalize,
) {
    match normalize {
        ConfusionMatrixNormalize::None => {}
        ConfusionMatrixNormalize::True => {
            for row in 0..classes {
                let offset = row * classes;
                let sum = data[offset..offset + classes].iter().sum::<f64>();
                if sum > f64::EPSILON {
                    for value in &mut data[offset..offset + classes] {
                        *value /= sum;
                    }
                }
            }
        }
        ConfusionMatrixNormalize::Pred => {
            for col in 0..classes {
                let sum = (0..classes)
                    .map(|row| data[row * classes + col])
                    .sum::<f64>();
                if sum > f64::EPSILON {
                    for row in 0..classes {
                        data[row * classes + col] /= sum;
                    }
                }
            }
        }
        ConfusionMatrixNormalize::All => {
            let sum = data.iter().sum::<f64>();
            if sum > f64::EPSILON {
                for value in data {
                    *value /= sum;
                }
            }
        }
    }
}
