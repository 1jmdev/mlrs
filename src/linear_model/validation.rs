use crate::darray::Array;

use super::LinearModelError;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedTargets {
    pub(crate) matrix: Array,
    pub(crate) is_vector: bool,
}

pub(crate) fn validate_features(x: &Array) -> Result<(), LinearModelError> {
    if !x.is_matrix() {
        return Err(LinearModelError::InvalidFeatureMatrixShape(
            x.shape().to_vec(),
        ));
    }

    Ok(())
}

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

pub(crate) fn format_coefficients(coefficients: &Array, is_vector: bool) -> Array {
    if is_vector {
        coefficients.column(0)
    } else {
        coefficients.transpose()
    }
}

pub(crate) fn format_intercepts(intercepts: &Array, is_vector: bool) -> Array {
    if is_vector {
        Array::scalar(intercepts.item())
    } else {
        intercepts.copy()
    }
}
