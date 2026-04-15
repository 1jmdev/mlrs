use crate::darray::Array;

use matrixmultiply::dgemm;
use rayon::prelude::*;
use wide::f64x4;

use super::LinearModelError;
use super::linear_regression::LinearRegression;
use super::validation::{
    format_coefficients, format_intercepts, prepare_targets, validate_features,
};

const SIMD_WIDTH: usize = 4;
const PAR_THRESHOLD: usize = 16_384;

impl LinearRegression {
    /// Fits the model to a feature matrix and target array.
    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, LinearModelError> {
        validate_features(x)?;
        let prepared_y = prepare_targets(x, y)?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        if n_samples == 0 || n_features == 0 {
            return Err(LinearModelError::EmptyInput);
        }
        if self.epochs == 0 {
            return Err(LinearModelError::InvalidEpochs(self.epochs));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(LinearModelError::InvalidLearningRate(self.learning_rate));
        }

        // Center the design matrix so the learned coefficients can be combined
        // with a stable post-hoc intercept term.
        let (x_used, x_offset) = if self.fit_intercept {
            let offset = x.mean_axis(0);
            let centered = x.sub_array(&offset.expand_dims(0));
            (centered, offset)
        } else {
            (x.copy(), Array::zeros(&[n_features]))
        };

        let n_targets = prepared_y.matrix.shape()[1];
        let (y_used, y_offset) = if self.fit_intercept {
            let offset = prepared_y.matrix.mean_axis(0);
            let centered = prepared_y.matrix.sub_array(&offset.expand_dims(0));
            (centered, offset)
        } else {
            (prepared_y.matrix.copy(), Array::zeros(&[n_targets]))
        };

        let coefficients =
            fit_coefficients_gradient_descent(&x_used, &y_used, self.epochs, self.learning_rate);

        // Recover the intercept in the original feature space after fitting on
        // centered data.
        let intercepts = if self.fit_intercept {
            let weighted_offsets = x_offset.expand_dims(0).matmul(&coefficients).squeeze();
            y_offset.sub_array(&weighted_offsets)
        } else {
            Array::zeros(&[n_targets])
        };

        self.n_features_in_ = Some(n_features);
        self.coef_ = Some(format_coefficients(&coefficients, prepared_y.is_vector));
        self.intercept_ = Some(format_intercepts(&intercepts, prepared_y.is_vector));

        Ok(self)
    }

    /// Returns observed targets minus model predictions.
    pub fn residuals(&self, x: &Array, y: &Array) -> Result<Array, LinearModelError> {
        let prediction = self.predict(x)?;
        let expected = if prediction.ndim() == 1 {
            y.copy()
        } else {
            prepare_targets(x, y)?.matrix
        };
        Ok(expected.sub_array(&prediction))
    }
}

/// Fits coefficient columns with batched gradient descent updates.
fn fit_coefficients_gradient_descent(
    x: &Array,
    y: &Array,
    epochs: usize,
    learning_rate: f64,
) -> Array {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let n_targets = y.shape()[1];
    let mut coefficients = vec![0.0; n_features * n_targets];
    let mut residuals = vec![0.0; n_samples * n_targets];
    let mut gradient = vec![0.0; n_features * n_targets];
    let step_size = learning_rate / n_samples as f64;

    for _ in 0..epochs {
        // Compute `X @ coefficients` into a reusable residual buffer.
        unsafe {
            dgemm(
                n_samples,
                n_features,
                n_targets,
                1.0,
                x.data.as_ptr(),
                n_features as isize,
                1,
                coefficients.as_ptr(),
                n_targets as isize,
                1,
                0.0,
                residuals.as_mut_ptr(),
                n_targets as isize,
                1,
            );
        }

        subtract_in_place(&mut residuals, &y.data);

        // Compute `X^T @ residuals` to obtain the full gradient matrix.
        unsafe {
            dgemm(
                n_features,
                n_samples,
                n_targets,
                1.0,
                x.data.as_ptr(),
                1,
                n_features as isize,
                residuals.as_ptr(),
                n_targets as isize,
                1,
                0.0,
                gradient.as_mut_ptr(),
                n_targets as isize,
                1,
            );
        }

        scaled_sub_assign(&mut coefficients, &gradient, step_size);
    }

    Array::from_shape_vec(&[n_features, n_targets], coefficients)
}

/// Subtracts one slice from another in place, parallelizing large inputs.
fn subtract_in_place(left: &mut [f64], right: &[f64]) {
    if left.len() >= PAR_THRESHOLD {
        left.par_chunks_mut(PAR_THRESHOLD)
            .zip(right.par_chunks(PAR_THRESHOLD))
            .for_each(|(left_chunk, right_chunk)| subtract_in_place_chunk(left_chunk, right_chunk));
    } else {
        subtract_in_place_chunk(left, right);
    }
}

/// Subtracts one chunk from another using SIMD where possible.
fn subtract_in_place_chunk(left: &mut [f64], right: &[f64]) {
    let simd_len = left.len() / SIMD_WIDTH * SIMD_WIDTH;

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
        let updated: [f64; SIMD_WIDTH] = (left_values - right_values).into();
        left[offset..offset + SIMD_WIDTH].copy_from_slice(&updated);
    }

    for index in simd_len..left.len() {
        left[index] -= right[index];
    }
}

/// Applies `left -= scale * right`, parallelizing large inputs.
fn scaled_sub_assign(left: &mut [f64], right: &[f64], scale: f64) {
    if left.len() >= PAR_THRESHOLD {
        left.par_chunks_mut(PAR_THRESHOLD)
            .zip(right.par_chunks(PAR_THRESHOLD))
            .for_each(|(left_chunk, right_chunk)| {
                scaled_sub_assign_chunk(left_chunk, right_chunk, scale)
            });
    } else {
        scaled_sub_assign_chunk(left, right, scale);
    }
}

/// Applies `left -= scale * right` to one chunk using SIMD where possible.
fn scaled_sub_assign_chunk(left: &mut [f64], right: &[f64], scale: f64) {
    let scale_values = f64x4::splat(scale);
    let simd_len = left.len() / SIMD_WIDTH * SIMD_WIDTH;

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
        let updated: [f64; SIMD_WIDTH] = (left_values - right_values * scale_values).into();
        left[offset..offset + SIMD_WIDTH].copy_from_slice(&updated);
    }

    for index in simd_len..left.len() {
        left[index] -= right[index] * scale;
    }
}
