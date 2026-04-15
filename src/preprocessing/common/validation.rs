use crate::darray::Array;

use super::super::PreprocessingError;

/// Validates that an input is a finite, non-empty feature matrix.
pub(crate) fn ensure_2d_finite(
    x: &Array,
    name: &'static str,
) -> Result<(usize, usize), PreprocessingError> {
    let shape = x.shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PreprocessingError::InvalidInputShape(shape.to_vec()));
    }
    if x.data().iter().any(|value| !value.is_finite()) {
        return Err(PreprocessingError::NonFiniteInput(name));
    }
    Ok((shape[0], shape[1]))
}

/// Validates that a label array is a finite, non-empty vector.
pub(crate) fn ensure_1d_finite(y: &Array, name: &'static str) -> Result<usize, PreprocessingError> {
    let shape = y.shape();
    if shape.len() != 1 || shape[0] == 0 {
        return Err(PreprocessingError::InvalidLabelShape(shape.to_vec()));
    }
    if y.data().iter().any(|value| !value.is_finite()) {
        return Err(PreprocessingError::NonFiniteInput(name));
    }
    Ok(shape[0])
}

/// Validates that runtime inputs use the fitted feature count.
pub(crate) fn ensure_feature_count(got: usize, expected: usize) -> Result<(), PreprocessingError> {
    if got == expected {
        return Ok(());
    }
    Err(PreprocessingError::FeatureCountMismatch { expected, got })
}

/// Validates a target range used by min-max scaling.
pub(crate) fn checked_feature_range(min: f64, max: f64) -> Result<(f64, f64), PreprocessingError> {
    if min.is_finite() && max.is_finite() && min < max {
        return Ok((min, max));
    }
    Err(PreprocessingError::InvalidFeatureRange { min, max })
}

/// Validates the robust scaling quantile interval in percentage space.
pub(crate) fn checked_quantile_range(
    lower: f64,
    upper: f64,
) -> Result<(f64, f64), PreprocessingError> {
    if lower.is_finite() && upper.is_finite() && 0.0 < lower && lower < upper && upper < 100.0 {
        return Ok((lower, upper));
    }
    Err(PreprocessingError::InvalidQuantileRange { lower, upper })
}

/// Matches sklearn's constant-feature handling by leaving zero scales at one.
pub(crate) fn is_effectively_zero(value: f64) -> bool {
    value.abs() <= f64::EPSILON
}
