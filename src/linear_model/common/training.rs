use crate::darray::Array;

use super::LinearModelError;
use super::validation::{PreparedTargets, format_coefficients, format_intercepts, prepare_targets};

pub(crate) struct TrainingData {
    pub(crate) x: Array,
    pub(crate) y: Array,
    pub(crate) x_offset: Array,
    pub(crate) y_offset: Array,
    pub(crate) n_features: usize,
    pub(crate) prepared_y: PreparedTargets,
}

pub(crate) fn prepare_training_data(
    x: &Array,
    y: &Array,
    fit_intercept: bool,
) -> Result<TrainingData, LinearModelError> {
    super::validation::validate_features(x)?;
    let prepared_y = prepare_targets(x, y)?;
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    if n_samples == 0 || n_features == 0 {
        return Err(LinearModelError::EmptyInput);
    }

    let n_targets = prepared_y.matrix.shape()[1];
    let (x_used, x_offset) = if fit_intercept {
        let offset = x.mean_axis(0);
        (x.sub_array(&offset.expand_dims(0)), offset)
    } else {
        (x.copy(), Array::zeros(&[n_features]))
    };
    let (y_used, y_offset) = if fit_intercept {
        let offset = prepared_y.matrix.mean_axis(0);
        (prepared_y.matrix.sub_array(&offset.expand_dims(0)), offset)
    } else {
        (prepared_y.matrix.copy(), Array::zeros(&[n_targets]))
    };

    Ok(TrainingData {
        x: x_used,
        y: y_used,
        x_offset,
        y_offset,
        n_features,
        prepared_y,
    })
}

pub(crate) fn finalize_parameters(
    coefficients: &Array,
    prepared_y: &PreparedTargets,
    x_offset: &Array,
    y_offset: &Array,
    fit_intercept: bool,
) -> Result<(Array, Array), LinearModelError> {
    let intercepts = if fit_intercept {
        let weighted_offsets = x_offset
            .expand_dims(0)
            .matmul(coefficients)
            .map_err(|_| LinearModelError::InvalidFeatureMatrixShape(x_offset.shape().to_vec()))?
            .squeeze();
        y_offset.sub_array(&weighted_offsets)
    } else {
        Array::zeros(&[y_offset.len()])
    };

    Ok((
        format_coefficients(coefficients, prepared_y.is_vector),
        format_intercepts(&intercepts, prepared_y.is_vector),
    ))
}
