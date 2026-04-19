use crate::darray::Array;

use matrixmultiply::dgemm;
use rayon::prelude::*;

use super::dot_simd;
use super::LinearModelError;
use super::validation::validate_features;

const PAR_THRESHOLD: usize = 16_384;

pub(crate) fn predict_from_parameters(
    x: &Array,
    coefficients: &Array,
    intercepts: &Array,
    expected_features: usize,
) -> Result<Array, LinearModelError> {
    validate_features(x)?;
    let got = x.shape()[1];
    if got != expected_features {
        return Err(LinearModelError::FeatureCountMismatch {
            expected: expected_features,
            got,
        });
    }

    Ok(if coefficients.ndim() == 1 {
        predict_single_target(x, coefficients, intercepts.item())
    } else {
        predict_multi_target(x, coefficients, intercepts)
    })
}

fn predict_single_target(x: &Array, coefficients: &Array, intercept: f64) -> Array {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut data = vec![0.0; rows];
    if rows * cols >= PAR_THRESHOLD {
        data.par_iter_mut().enumerate().for_each(|(row, value)| {
            let start = row * cols;
            *value = dot_simd(&x.data[start..start + cols], &coefficients.data) + intercept;
        });
    } else {
        for (row, value) in data.iter_mut().enumerate() {
            let start = row * cols;
            *value = dot_simd(&x.data[start..start + cols], &coefficients.data) + intercept;
        }
    }
    Array::from_shape_vec(&[rows], data)
}

fn predict_multi_target(x: &Array, coefficients: &Array, intercepts: &Array) -> Array {
    let rows = x.shape()[0];
    let shared = x.shape()[1];
    let cols = coefficients.shape()[0];
    let mut data = vec![0.0; rows * cols];
    unsafe {
        dgemm(
            rows,
            shared,
            cols,
            1.0,
            x.data.as_ptr(),
            shared as isize,
            1,
            coefficients.data.as_ptr(),
            1,
            shared as isize,
            0.0,
            data.as_mut_ptr(),
            cols as isize,
            1,
        );
    }
    if data.len() >= PAR_THRESHOLD {
        data.par_chunks_mut(cols).for_each(|row| {
            for (value, intercept) in row.iter_mut().zip(intercepts.data()) {
                *value += intercept;
            }
        });
    } else {
        for row in data.chunks_mut(cols) {
            for (value, intercept) in row.iter_mut().zip(intercepts.data()) {
                *value += intercept;
            }
        }
    }
    Array::from_shape_vec(&[rows, cols], data)
}
