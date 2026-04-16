use crate::darray::Array;

use matrixmultiply::dgemm;
use rayon::prelude::*;
use wide::f64x4;

use super::LinearModelError;
use super::validation::validate_features;

const SIMD_WIDTH: usize = 4;
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

fn dot_simd(left: &[f64], right: &[f64]) -> f64 {
    let simd_len = left.len() / SIMD_WIDTH * SIMD_WIDTH;
    let mut accum = f64x4::splat(0.0);
    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        accum += f64x4::from([
            left[offset],
            left[offset + 1],
            left[offset + 2],
            left[offset + 3],
        ]) * f64x4::from([
            right[offset],
            right[offset + 1],
            right[offset + 2],
            right[offset + 3],
        ]);
    }
    let partials: [f64; SIMD_WIDTH] = accum.into();
    partials.into_iter().sum::<f64>()
        + left[simd_len..]
            .iter()
            .zip(&right[simd_len..])
            .map(|(left_value, right_value)| left_value * right_value)
            .sum::<f64>()
}
