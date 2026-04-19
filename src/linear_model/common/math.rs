use crate::darray::Array;

use matrixmultiply::dgemm;
use rayon::prelude::*;
use wide::f64x4;

use super::LinearModelError;

const SIMD_WIDTH: usize = 4;
const PAR_THRESHOLD: usize = 16_384;

pub(crate) fn compute_gram_and_xty(x: &Array, y: &Array) -> (Vec<f64>, Vec<f64>) {
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

    (gram, xty)
}

pub(crate) fn dot_simd(left: &[f64], right: &[f64]) -> f64 {
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

pub(crate) fn cholesky_decompose(matrix: &[f64], size: usize) -> Result<Vec<f64>, LinearModelError> {
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

pub(crate) fn cholesky_solve(lower: &[f64], rhs: &[f64], size: usize) -> Vec<f64> {
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

pub(crate) fn subtract_in_place(left: &mut [f64], right: &[f64]) {
    if left.len() >= PAR_THRESHOLD {
        left.par_chunks_mut(PAR_THRESHOLD)
            .zip(right.par_chunks(PAR_THRESHOLD))
            .for_each(|(left_chunk, right_chunk)| subtract_in_place_chunk(left_chunk, right_chunk));
    } else {
        subtract_in_place_chunk(left, right);
    }
}

pub(crate) fn scaled_sub_assign(left: &mut [f64], right: &[f64], scale: f64) {
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

pub(crate) fn scaled_add_assign(left: &mut [f64], right: &[f64], scale: f64) {
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

pub(crate) fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
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
