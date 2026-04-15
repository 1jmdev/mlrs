use crate::darray::Array;

use rayon::prelude::*;
use wide::f64x4;

const SIMD_WIDTH: usize = 4;
const PAR_THRESHOLD: usize = 16_384;

pub(crate) fn r2_score_1d(y_true: &Array, y_pred: &Array) -> f64 {
    let mean = y_true.mean();
    let residual_sum = sum_squared_differences(&y_true.data, &y_pred.data);
    let total_sum = sum_squared_centered(&y_true.data, mean);

    if total_sum <= f64::EPSILON {
        if residual_sum <= f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - residual_sum / total_sum
    }
}

pub(crate) fn mean_r2_score_matrix(y_true: &Array, y_pred: &Array) -> f64 {
    let rows = y_true.shape()[0];
    let cols = y_true.shape()[1];
    let total: f64 = if rows * cols >= PAR_THRESHOLD {
        (0..cols)
            .into_par_iter()
            .map(|column| r2_score_column(y_true, y_pred, column, rows, cols))
            .sum()
    } else {
        (0..cols)
            .map(|column| r2_score_column(y_true, y_pred, column, rows, cols))
            .sum()
    };

    total / cols as f64
}

fn r2_score_column(y_true: &Array, y_pred: &Array, column: usize, rows: usize, cols: usize) -> f64 {
    let mean = (0..rows)
        .map(|row| y_true.data[row * cols + column])
        .sum::<f64>()
        / rows as f64;

    let residual_sum = (0..rows)
        .map(|row| {
            let offset = row * cols + column;
            let delta = y_true.data[offset] - y_pred.data[offset];
            delta * delta
        })
        .sum::<f64>();
    let total_sum = (0..rows)
        .map(|row| {
            let delta = y_true.data[row * cols + column] - mean;
            delta * delta
        })
        .sum::<f64>();

    if total_sum <= f64::EPSILON {
        if residual_sum <= f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - residual_sum / total_sum
    }
}

fn sum_squared_differences(left: &[f64], right: &[f64]) -> f64 {
    if left.len() >= PAR_THRESHOLD {
        left.par_chunks(PAR_THRESHOLD)
            .zip(right.par_chunks(PAR_THRESHOLD))
            .map(|(left_chunk, right_chunk)| sum_squared_differences_chunk(left_chunk, right_chunk))
            .sum()
    } else {
        sum_squared_differences_chunk(left, right)
    }
}

fn sum_squared_differences_chunk(left: &[f64], right: &[f64]) -> f64 {
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
        let delta = left_values - right_values;
        accum += delta * delta;
    }

    let partials: [f64; SIMD_WIDTH] = accum.into();
    let mut total = partials.into_iter().sum::<f64>();
    total += left[simd_len..]
        .iter()
        .zip(&right[simd_len..])
        .map(|(left_value, right_value)| {
            let delta = left_value - right_value;
            delta * delta
        })
        .sum::<f64>();
    total
}

fn sum_squared_centered(values: &[f64], mean: f64) -> f64 {
    if values.len() >= PAR_THRESHOLD {
        values
            .par_chunks(PAR_THRESHOLD)
            .map(|chunk| sum_squared_centered_chunk(chunk, mean))
            .sum()
    } else {
        sum_squared_centered_chunk(values, mean)
    }
}

fn sum_squared_centered_chunk(values: &[f64], mean: f64) -> f64 {
    let mean_values = f64x4::splat(mean);
    let simd_len = values.len() / SIMD_WIDTH * SIMD_WIDTH;
    let mut accum = f64x4::splat(0.0);

    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let current = f64x4::from([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let delta = current - mean_values;
        accum += delta * delta;
    }

    let partials: [f64; SIMD_WIDTH] = accum.into();
    let mut total = partials.into_iter().sum::<f64>();
    total += values[simd_len..]
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>();
    total
}
