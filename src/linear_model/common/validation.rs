use crate::darray::Array;

use super::LinearModelError;

/// Stores validated targets in a uniform matrix representation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedTargets {
    /// Holds the target data as a two-dimensional array.
    pub(crate) matrix: Array,
    /// Records whether the original input was a one-dimensional vector.
    pub(crate) is_vector: bool,
}

/// Stores validated class labels and their encoded indices.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedClasses {
    pub(crate) classes: Vec<f64>,
    pub(crate) encoded: Vec<usize>,
}

/// Validates that the feature input is a two-dimensional matrix.
pub(crate) fn validate_features(x: &Array) -> Result<(), LinearModelError> {
    if !x.is_matrix() {
        return Err(LinearModelError::InvalidFeatureMatrixShape(
            x.shape().to_vec(),
        ));
    }

    Ok(())
}

/// Normalizes targets into a matrix while preserving vector-vs-matrix shape.
pub(crate) fn prepare_targets(x: &Array, y: &Array) -> Result<PreparedTargets, LinearModelError> {
    let x_samples = x.shape()[0];

    match y.ndim() {
        1 => {
            if y.len() != x_samples {
                return Err(LinearModelError::SampleCountMismatch {
                    x_samples,
                    y_samples: y.len(),
                });
            }

            Ok(PreparedTargets {
                matrix: y.expand_dims(1),
                is_vector: true,
            })
        }
        2 => {
            if y.shape()[0] != x_samples {
                return Err(LinearModelError::SampleCountMismatch {
                    x_samples,
                    y_samples: y.shape()[0],
                });
            }

            Ok(PreparedTargets {
                matrix: y.copy(),
                is_vector: false,
            })
        }
        _ => Err(LinearModelError::InvalidTargetShape(y.shape().to_vec())),
    }
}

/// Validates a one-dimensional label vector and encodes distinct classes.
pub(crate) fn prepare_class_labels(
    x: &Array,
    y: &Array,
) -> Result<PreparedClasses, LinearModelError> {
    if !y.is_vector() || y.is_empty() {
        return Err(LinearModelError::InvalidLabelShape(y.shape().to_vec()));
    }
    if y.len() != x.shape()[0] {
        return Err(LinearModelError::SampleCountMismatch {
            x_samples: x.shape()[0],
            y_samples: y.len(),
        });
    }
    if y.data().iter().any(|value| !value.is_finite()) {
        return Err(LinearModelError::InvalidLabelShape(y.shape().to_vec()));
    }

    let mut classes = y.to_vec();
    classes.sort_by(f64::total_cmp);
    classes.dedup_by(|left, right| left.total_cmp(right).is_eq());
    if classes.len() < 2 {
        return Err(LinearModelError::InvalidClassCount(classes.len()));
    }

    let encoded = y
        .data()
        .iter()
        .map(|value| {
            classes
                .binary_search_by(|class| class.total_cmp(value))
                .map_err(|_| LinearModelError::InvalidLabelShape(y.shape().to_vec()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PreparedClasses { classes, encoded })
}

/// Formats learned coefficients to match the original target dimensionality.
pub(crate) fn format_coefficients(coefficients: &Array, is_vector: bool) -> Array {
    if is_vector {
        coefficients.column(0)
    } else {
        coefficients.transpose()
    }
}

/// Formats learned intercepts to match the original target dimensionality.
pub(crate) fn format_intercepts(intercepts: &Array, is_vector: bool) -> Array {
    if is_vector {
        Array::scalar(intercepts.item())
    } else {
        intercepts.copy()
    }
}
