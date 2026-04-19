use super::super::Array;
use super::layout::{PAR_CHUNK_LEN, PAR_THRESHOLD};
use rayon::prelude::*;
use wide::f64x4;

const SIMD_WIDTH: usize = 4;

#[inline]
fn load_f64x4(values: &[f64]) -> f64x4 {
    f64x4::from([values[0], values[1], values[2], values[3]])
}

#[inline]
fn store_f64x4(output: &mut [f64], values: f64x4) {
    let array: [f64; SIMD_WIDTH] = values.into();
    output[..SIMD_WIDTH].copy_from_slice(&array);
}

pub(crate) fn unary_map_simd<FScalar, FVec>(
    array: &Array,
    scalar_op: FScalar,
    vec_op: FVec,
) -> Array
where
    FScalar: Fn(f64) -> f64 + Copy + Send + Sync,
    FVec: Fn(f64x4) -> f64x4 + Copy + Send + Sync,
{
    let mut data = vec![0.0; array.data.len()];

    if array.data.len() >= PAR_THRESHOLD {
        data.par_chunks_mut(PAR_CHUNK_LEN)
            .enumerate()
            .for_each(|(chunk_index, output)| {
                let start = chunk_index * PAR_CHUNK_LEN;
                let input = &array.data[start..start + output.len()];
                unary_map_simd_chunk(input, output, scalar_op, vec_op);
            });
    } else {
        unary_map_simd_chunk(&array.data, &mut data, scalar_op, vec_op);
    }

    Array {
        data,
        shape: array.shape.clone(),
        strides: array.strides.clone(),
    }
}

fn unary_map_simd_chunk<FScalar, FVec>(
    input: &[f64],
    output: &mut [f64],
    scalar_op: FScalar,
    vec_op: FVec,
) where
    FScalar: Fn(f64) -> f64 + Copy,
    FVec: Fn(f64x4) -> f64x4 + Copy,
{
    let simd_len = input.len() / SIMD_WIDTH * SIMD_WIDTH;

    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let values = load_f64x4(&input[offset..offset + SIMD_WIDTH]);
        store_f64x4(&mut output[offset..offset + SIMD_WIDTH], vec_op(values));
    }

    for index in simd_len..input.len() {
        output[index] = scalar_op(input[index]);
    }
}

pub(crate) fn binary_map_same_shape_simd<FScalar, FVec>(
    left: &Array,
    right: &Array,
    scalar_op: FScalar,
    vec_op: FVec,
) -> Array
where
    FScalar: Fn(f64, f64) -> f64 + Copy + Send + Sync,
    FVec: Fn(f64x4, f64x4) -> f64x4 + Copy + Send + Sync,
{
    let mut data = vec![0.0; left.data.len()];

    if left.data.len() >= PAR_THRESHOLD {
        data.par_chunks_mut(PAR_CHUNK_LEN)
            .enumerate()
            .for_each(|(chunk_index, output)| {
                let start = chunk_index * PAR_CHUNK_LEN;
                let left_input = &left.data[start..start + output.len()];
                let right_input = &right.data[start..start + output.len()];
                binary_map_same_shape_simd_chunk(
                    left_input,
                    right_input,
                    output,
                    scalar_op,
                    vec_op,
                );
            });
    } else {
        binary_map_same_shape_simd_chunk(&left.data, &right.data, &mut data, scalar_op, vec_op);
    }

    Array {
        data,
        shape: left.shape.clone(),
        strides: left.strides.clone(),
    }
}

fn binary_map_same_shape_simd_chunk<FScalar, FVec>(
    left: &[f64],
    right: &[f64],
    output: &mut [f64],
    scalar_op: FScalar,
    vec_op: FVec,
) where
    FScalar: Fn(f64, f64) -> f64 + Copy,
    FVec: Fn(f64x4, f64x4) -> f64x4 + Copy,
{
    let simd_len = output.len() / SIMD_WIDTH * SIMD_WIDTH;

    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let left_values = load_f64x4(&left[offset..offset + SIMD_WIDTH]);
        let right_values = load_f64x4(&right[offset..offset + SIMD_WIDTH]);
        store_f64x4(
            &mut output[offset..offset + SIMD_WIDTH],
            vec_op(left_values, right_values),
        );
    }

    for index in simd_len..output.len() {
        output[index] = scalar_op(left[index], right[index]);
    }
}

pub(crate) fn sum_simd(values: &[f64]) -> f64 {
    if values.len() >= PAR_THRESHOLD {
        values.par_chunks(PAR_CHUNK_LEN).map(sum_simd_chunk).sum()
    } else {
        sum_simd_chunk(values)
    }
}

pub(crate) fn clone_data_parallel(values: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; values.len()];
    if values.len() >= PAR_THRESHOLD {
        output
            .par_chunks_mut(PAR_CHUNK_LEN)
            .zip(values.par_chunks(PAR_CHUNK_LEN))
            .for_each(|(out, input)| out.copy_from_slice(input));
    } else {
        output.copy_from_slice(values);
    }
    output
}

fn sum_simd_chunk(values: &[f64]) -> f64 {
    let simd_len = values.len() / SIMD_WIDTH * SIMD_WIDTH;
    let mut accum = f64x4::splat(0.0);

    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        accum += load_f64x4(&values[offset..offset + SIMD_WIDTH]);
    }

    let mut total = 0.0;
    let partials: [f64; SIMD_WIDTH] = accum.into();
    for value in partials {
        total += value;
    }
    total + values[simd_len..].iter().sum::<f64>()
}

pub(crate) fn dot_simd(left: &[f64], right: &[f64]) -> f64 {
    if left.len() >= PAR_THRESHOLD {
        left.par_chunks(PAR_CHUNK_LEN)
            .zip(right.par_chunks(PAR_CHUNK_LEN))
            .map(|(left_chunk, right_chunk)| dot_simd_chunk(left_chunk, right_chunk))
            .sum()
    } else {
        dot_simd_chunk(left, right)
    }
}

fn dot_simd_chunk(left: &[f64], right: &[f64]) -> f64 {
    let simd_len = left.len() / SIMD_WIDTH * SIMD_WIDTH;
    let mut accum = f64x4::splat(0.0);

    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let left_values = load_f64x4(&left[offset..offset + SIMD_WIDTH]);
        let right_values = load_f64x4(&right[offset..offset + SIMD_WIDTH]);
        accum += left_values * right_values;
    }

    let mut total = 0.0;
    let partials: [f64; SIMD_WIDTH] = accum.into();
    for value in partials {
        total += value;
    }
    total
        + left[simd_len..]
            .iter()
            .zip(&right[simd_len..])
            .map(|(left_value, right_value)| left_value * right_value)
            .sum::<f64>()
}
