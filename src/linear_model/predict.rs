use crate::darray::Array;
use crate::metrics;

use matrixmultiply::dgemm;
use rayon::prelude::*;
use wide::f64x4;

use super::LinearModelError;
use super::linear_regression::LinearRegression;
use super::validation::validate_features;

const SIMD_WIDTH: usize = 4;
const PAR_THRESHOLD: usize = 16_384;

impl LinearRegression {
    pub fn predict(&self, x: &Array) -> Result<Array, LinearModelError> {
        validate_features(x)?;
        let expected = self.n_features_in_.ok_or(LinearModelError::NotFitted)?;
        let got = x.shape()[1];
        if got != expected {
            return Err(LinearModelError::FeatureCountMismatch { expected, got });
        }

        let coefficients = self.coef_.as_ref().ok_or(LinearModelError::NotFitted)?;
        let intercepts = self
            .intercept_
            .as_ref()
            .ok_or(LinearModelError::NotFitted)?;

        let prediction = if coefficients.ndim() == 1 {
            predict_single_target(x, coefficients, intercepts.item())
        } else {
            predict_multi_target(x, coefficients, intercepts)
        };

        Ok(prediction)
    }

    pub fn score(&self, x: &Array, y: &Array) -> Result<f64, LinearModelError> {
        let prediction = self.predict(x)?;
        metrics::r2_score(y, &prediction)
            .map_err(|_| LinearModelError::InvalidTargetShape(y.shape().to_vec()))
    }
}

fn predict_single_target(x: &Array, coefficients: &Array, intercept: f64) -> Array {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut data = vec![0.0; rows];

    if rows * cols >= PAR_THRESHOLD {
        data.par_iter_mut().enumerate().for_each(|(row, value)| {
            let start = row * cols;
            let end = start + cols;
            *value = dot_simd(&x.data[start..end], &coefficients.data) + intercept;
        });
    } else {
        for (row, value) in data.iter_mut().enumerate() {
            let start = row * cols;
            let end = start + cols;
            *value = dot_simd(&x.data[start..end], &coefficients.data) + intercept;
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
            for (value, intercept) in row.iter_mut().zip(&intercepts.data) {
                *value += intercept;
            }
        });
    } else {
        for row in data.chunks_mut(cols) {
            for (value, intercept) in row.iter_mut().zip(&intercepts.data) {
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
        let left_values = f64x4::from([
            left[offset],
            left[offset + 1],
            left[offset + 2],
            left[offset + 3],
        ]);
        let right_values = f64x4::from([
            right[offset],
            right[offset + 1],
            right[offset + 2],
            right[offset + 3],
        ]);
        accum += left_values * right_values;
    }

    let partials: [f64; SIMD_WIDTH] = accum.into();
    let mut total = partials.into_iter().sum::<f64>();
    total += left[simd_len..]
        .iter()
        .zip(&right[simd_len..])
        .map(|(left_value, right_value)| left_value * right_value)
        .sum::<f64>();
    total
}
