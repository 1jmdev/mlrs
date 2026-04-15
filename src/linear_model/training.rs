use crate::darray::Array;

use matrixmultiply::dgemm;
use rayon::prelude::*;
use wide::f64x4;

use super::validation::{format_coefficients, format_intercepts, prepare_targets, PreparedTargets};
use super::LinearModelError;

const SIMD_WIDTH: usize = 4;
const PAR_THRESHOLD: usize = 16_384;

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

pub(crate) fn fit_linear_coefficients(
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

pub(crate) fn fit_ridge_coefficients(
    x: &Array,
    y: &Array,
    alpha: f64,
) -> Result<Array, LinearModelError> {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let n_targets = y.shape()[1];
    let mut gram = vec![0.0; n_features * n_features];
    let mut xty = vec![0.0; n_features * n_targets];

    unsafe {
        dgemm(
            n_features,
            n_samples,
            n_features,
            1.0,
            x.data.as_ptr(),
            1,
            n_features as isize,
            x.data.as_ptr(),
            n_features as isize,
            1,
            0.0,
            gram.as_mut_ptr(),
            n_features as isize,
            1,
        );
        dgemm(
            n_features,
            n_samples,
            n_targets,
            1.0,
            x.data.as_ptr(),
            1,
            n_features as isize,
            y.data.as_ptr(),
            n_targets as isize,
            1,
            0.0,
            xty.as_mut_ptr(),
            n_targets as isize,
            1,
        );
    }

    for feature in 0..n_features {
        gram[feature * n_features + feature] += alpha;
    }

    let cholesky = cholesky_decompose(&gram, n_features)?;
    let mut coefficients = vec![0.0; n_features * n_targets];
    for target in 0..n_targets {
        let rhs = (0..n_features)
            .map(|feature| xty[feature * n_targets + target])
            .collect::<Vec<_>>();
        let solution = cholesky_solve(&cholesky, &rhs, n_features);
        for (feature, value) in solution.into_iter().enumerate() {
            coefficients[feature * n_targets + target] = value;
        }
    }

    Ok(Array::from_shape_vec(
        &[n_features, n_targets],
        coefficients,
    ))
}

pub(crate) fn fit_lasso_coefficients(
    x: &Array,
    y: &Array,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Array {
    let n_features = x.shape()[1];
    let n_targets = y.shape()[1];
    let n_samples = x.shape()[0];
    let mut gram = vec![0.0; n_features * n_features];
    let mut xty = vec![0.0; n_features * n_targets];

    unsafe {
        dgemm(
            n_features,
            n_samples,
            n_features,
            1.0,
            x.data.as_ptr(),
            1,
            n_features as isize,
            x.data.as_ptr(),
            n_features as isize,
            1,
            0.0,
            gram.as_mut_ptr(),
            n_features as isize,
            1,
        );
        dgemm(
            n_features,
            n_samples,
            n_targets,
            1.0,
            x.data.as_ptr(),
            1,
            n_features as isize,
            y.data.as_ptr(),
            n_targets as isize,
            1,
            0.0,
            xty.as_mut_ptr(),
            n_targets as isize,
            1,
        );
    }

    let feature_norms = (0..n_features)
        .map(|feature| gram[feature * n_features + feature])
        .collect::<Vec<_>>();
    let coefficients = (0..n_targets)
        .into_par_iter()
        .map(|target| {
            let target_xty = (0..n_features)
                .map(|feature| xty[feature * n_targets + target])
                .collect::<Vec<_>>();
            fit_lasso_target(
                &gram,
                &target_xty,
                &feature_norms,
                alpha,
                max_iter,
                tol,
                n_features,
                n_samples,
            )
        })
        .collect::<Vec<_>>();

    let mut data = vec![0.0; n_features * n_targets];
    for (target, values) in coefficients.into_iter().enumerate() {
        for (feature, value) in values.into_iter().enumerate() {
            data[feature * n_targets + target] = value;
        }
    }
    Array::from_shape_vec(&[n_features, n_targets], data)
}

pub(crate) fn finalize_parameters(
    coefficients: &Array,
    prepared_y: &PreparedTargets,
    x_offset: &Array,
    y_offset: &Array,
    fit_intercept: bool,
) -> (Array, Array) {
    let intercepts = if fit_intercept {
        let weighted_offsets = x_offset.expand_dims(0).matmul(coefficients).squeeze();
        y_offset.sub_array(&weighted_offsets)
    } else {
        Array::zeros(&[y_offset.len()])
    };

    (
        format_coefficients(coefficients, prepared_y.is_vector),
        format_intercepts(&intercepts, prepared_y.is_vector),
    )
}

fn fit_lasso_target(
    gram: &[f64],
    xty: &[f64],
    feature_norms: &[f64],
    alpha: f64,
    max_iter: usize,
    tol: f64,
    n_features: usize,
    n_samples: usize,
) -> Vec<f64> {
    let mut coefficients = vec![0.0; n_features];
    let mut correlations = vec![0.0; n_features];
    let penalty = alpha * n_samples as f64;

    for _ in 0..max_iter {
        let mut max_update = 0.0_f64;
        for feature in 0..n_features {
            let norm = feature_norms[feature];
            if norm <= f64::EPSILON {
                continue;
            }

            let old = coefficients[feature];
            let rho = xty[feature] - (correlations[feature] - norm * old);

            let updated = soft_threshold(rho, penalty) / norm;
            let delta = updated - old;
            if delta != 0.0 {
                coefficients[feature] = updated;
                let row = &gram[feature * n_features..(feature + 1) * n_features];
                scaled_add_assign(&mut correlations, row, delta);
                max_update = max_update.max(delta.abs());
            }
        }
        if max_update <= tol {
            break;
        }
    }

    coefficients
}

fn cholesky_decompose(matrix: &[f64], size: usize) -> Result<Vec<f64>, LinearModelError> {
    let mut lower = vec![0.0; matrix.len()];
    for row in 0..size {
        for col in 0..=row {
            let mut sum = matrix[row * size + col];
            for inner in 0..col {
                sum -= lower[row * size + inner] * lower[col * size + inner];
            }
            if row == col {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err(LinearModelError::SingularMatrix);
                }
                lower[row * size + col] = sum.sqrt();
            } else {
                lower[row * size + col] = sum / lower[col * size + col];
            }
        }
    }
    Ok(lower)
}

fn cholesky_solve(lower: &[f64], rhs: &[f64], size: usize) -> Vec<f64> {
    let mut y = vec![0.0; size];
    for row in 0..size {
        let mut sum = rhs[row];
        for col in 0..row {
            sum -= lower[row * size + col] * y[col];
        }
        y[row] = sum / lower[row * size + row];
    }

    let mut x = vec![0.0; size];
    for row in (0..size).rev() {
        let mut sum = y[row];
        for col in row + 1..size {
            sum -= lower[col * size + row] * x[col];
        }
        x[row] = sum / lower[row * size + row];
    }
    x
}

fn subtract_in_place(left: &mut [f64], right: &[f64]) {
    if left.len() >= PAR_THRESHOLD {
        left.par_chunks_mut(PAR_THRESHOLD)
            .zip(right.par_chunks(PAR_THRESHOLD))
            .for_each(|(left_chunk, right_chunk)| subtract_in_place_chunk(left_chunk, right_chunk));
    } else {
        subtract_in_place_chunk(left, right);
    }
}

fn subtract_in_place_chunk(left: &mut [f64], right: &[f64]) {
    let simd_len = left.len() / SIMD_WIDTH * SIMD_WIDTH;
    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let updated: [f64; SIMD_WIDTH] = (f64x4::from([
            left[offset],
            left[offset + 1],
            left[offset + 2],
            left[offset + 3],
        ]) - f64x4::from([
            right[offset],
            right[offset + 1],
            right[offset + 2],
            right[offset + 3],
        ]))
        .into();
        left[offset..offset + SIMD_WIDTH].copy_from_slice(&updated);
    }
    for index in simd_len..left.len() {
        left[index] -= right[index];
    }
}

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

fn scaled_sub_assign_chunk(left: &mut [f64], right: &[f64], scale: f64) {
    let scale_values = f64x4::splat(scale);
    let simd_len = left.len() / SIMD_WIDTH * SIMD_WIDTH;
    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let updated: [f64; SIMD_WIDTH] = (f64x4::from([
            left[offset],
            left[offset + 1],
            left[offset + 2],
            left[offset + 3],
        ]) - f64x4::from([
            right[offset],
            right[offset + 1],
            right[offset + 2],
            right[offset + 3],
        ]) * scale_values)
            .into();
        left[offset..offset + SIMD_WIDTH].copy_from_slice(&updated);
    }
    for index in simd_len..left.len() {
        left[index] -= right[index] * scale;
    }
}

fn scaled_add_assign(left: &mut [f64], right: &[f64], scale: f64) {
    let scale_values = f64x4::splat(scale);
    let simd_len = left.len() / SIMD_WIDTH * SIMD_WIDTH;
    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let updated: [f64; SIMD_WIDTH] = (f64x4::from([
            left[offset],
            left[offset + 1],
            left[offset + 2],
            left[offset + 3],
        ]) + f64x4::from([
            right[offset],
            right[offset + 1],
            right[offset + 2],
            right[offset + 3],
        ]) * scale_values)
            .into();
        left[offset..offset + SIMD_WIDTH].copy_from_slice(&updated);
    }
    for index in simd_len..left.len() {
        left[index] += right[index] * scale;
    }
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}
