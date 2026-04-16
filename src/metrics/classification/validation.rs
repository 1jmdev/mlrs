use crate::darray::Array;

use super::super::MetricsError;

pub(crate) fn validate_label_vectors(
    y_true: &Array,
    y_pred: &Array,
) -> Result<usize, MetricsError> {
    if !y_true.is_vector() {
        return Err(MetricsError::InvalidClassificationShape(
            y_true.shape().to_vec(),
        ));
    }
    if !y_pred.is_vector() {
        return Err(MetricsError::InvalidClassificationShape(
            y_pred.shape().to_vec(),
        ));
    }
    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            y_true: y_true.shape().to_vec(),
            y_pred: y_pred.shape().to_vec(),
        });
    }
    if y_true.data().iter().any(|value| !value.is_finite()) {
        return Err(MetricsError::NonFiniteInput("y_true"));
    }
    if y_pred.data().iter().any(|value| !value.is_finite()) {
        return Err(MetricsError::NonFiniteInput("y_pred"));
    }
    Ok(y_true.len())
}

pub(crate) fn validate_sample_weight(
    sample_weight: Option<&Array>,
    expected: usize,
) -> Result<Option<&[f64]>, MetricsError> {
    let Some(sample_weight) = sample_weight else {
        return Ok(None);
    };
    if !sample_weight.is_vector() {
        return Err(MetricsError::InvalidSampleWeightShape(
            sample_weight.shape().to_vec(),
        ));
    }
    if sample_weight.len() != expected {
        return Err(MetricsError::SampleWeightLengthMismatch {
            expected,
            got: sample_weight.len(),
        });
    }
    if sample_weight.data().iter().any(|value| !value.is_finite()) {
        return Err(MetricsError::NonFiniteInput("sample_weight"));
    }
    Ok(Some(sample_weight.data()))
}

pub(crate) fn resolve_labels(
    y_true: &Array,
    y_pred: &Array,
    labels: Option<&Array>,
) -> Result<Vec<f64>, MetricsError> {
    if let Some(labels) = labels {
        if !labels.is_vector() {
            return Err(MetricsError::InvalidClassificationShape(
                labels.shape().to_vec(),
            ));
        }
        if labels.is_empty() {
            return Err(MetricsError::EmptyLabels);
        }
        if labels.data().iter().any(|value| !value.is_finite()) {
            return Err(MetricsError::NonFiniteInput("labels"));
        }
        let mut resolved: Vec<f64> = Vec::with_capacity(labels.len());
        for &label in labels.data() {
            if label_index(&resolved, label).is_none() {
                resolved.push(label);
            }
        }
        if resolved.is_empty() {
            return Err(MetricsError::EmptyLabels);
        }
        return Ok(resolved);
    }

    let mut resolved = y_true.to_vec();
    resolved.extend_from_slice(y_pred.data());
    resolved.sort_by(f64::total_cmp);
    resolved.dedup_by(|left, right| left.total_cmp(right).is_eq());
    if resolved.is_empty() {
        Err(MetricsError::EmptyLabels)
    } else {
        Ok(resolved)
    }
}

pub(crate) fn checked_zero_division(zero_division: f64) -> Result<f64, MetricsError> {
    if zero_division.is_finite() {
        Ok(zero_division)
    } else {
        Err(MetricsError::InvalidZeroDivision(zero_division))
    }
}

pub(crate) fn label_index(labels: &[f64], value: f64) -> Option<usize> {
    labels
        .iter()
        .position(|label: &f64| label.total_cmp(&value).is_eq())
}
