use crate::darray::Array;

use super::TreeError;

pub(crate) fn validate_fit_inputs(x: &Array, y: &Array) -> Result<(), TreeError> {
    if !x.is_matrix() {
        return Err(TreeError::InvalidFeatureMatrixShape(x.shape().to_vec()));
    }
    if x.shape()[0] == 0 || x.shape()[1] == 0 {
        return Err(TreeError::EmptyInput);
    }
    if !y.is_vector() || y.is_empty() {
        return Err(TreeError::InvalidLabelShape(y.shape().to_vec()));
    }
    if x.shape()[0] != y.len() {
        return Err(TreeError::SampleCountMismatch {
            x_samples: x.shape()[0],
            y_samples: y.len(),
        });
    }
    Ok(())
}

pub(crate) fn validate_predict_input(x: &Array, expected: usize) -> Result<(), TreeError> {
    if !x.is_matrix() {
        return Err(TreeError::InvalidFeatureMatrixShape(x.shape().to_vec()));
    }
    if x.shape()[1] != expected {
        return Err(TreeError::FeatureCountMismatch {
            expected,
            got: x.shape()[1],
        });
    }
    Ok(())
}
