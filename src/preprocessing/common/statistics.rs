use crate::darray::Array;

use super::{SIMD_LANES, load_f64x4, store_f64x4};
use wide::f64x4;

/// Computes per-feature means and population variances.
pub(crate) fn column_mean_var(x: &Array, rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let data = x.data();
    let mut means = vec![0.0; cols];
    for row in 0..rows {
        let offset = row * cols;
        let row_values = &data[offset..offset + cols];
        let mut col = 0;
        while col + SIMD_LANES <= cols {
            let accum = load_f64x4(&means, col) + load_f64x4(row_values, col);
            store_f64x4(&mut means, col, accum);
            col += SIMD_LANES;
        }
        while col < cols {
            means[col] += row_values[col];
            col += 1;
        }
    }
    for mean in &mut means {
        *mean /= rows as f64;
    }

    let mut variances = vec![0.0; cols];
    for row in 0..rows {
        let offset = row * cols;
        let row_values = &data[offset..offset + cols];
        let mut col = 0;
        while col + SIMD_LANES <= cols {
            let centered = load_f64x4(row_values, col) - load_f64x4(&means, col);
            let accum = load_f64x4(&variances, col) + centered * centered;
            store_f64x4(&mut variances, col, accum);
            col += SIMD_LANES;
        }
        while col < cols {
            let centered = row_values[col] - means[col];
            variances[col] += centered * centered;
            col += 1;
        }
    }
    for variance in &mut variances {
        *variance /= rows as f64;
    }

    (means, variances)
}

/// Computes per-feature minima and maxima.
pub(crate) fn column_min_max(x: &Array, rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let data = x.data();
    let mut mins = vec![f64::INFINITY; cols];
    let mut maxs = vec![f64::NEG_INFINITY; cols];

    for row in 0..rows {
        let offset = row * cols;
        let row_values = &data[offset..offset + cols];
        let mut col = 0;
        for ((min_chunk, max_chunk), value_chunk) in mins
            .chunks_exact_mut(SIMD_LANES)
            .zip(maxs.chunks_exact_mut(SIMD_LANES))
            .zip(row_values.chunks_exact(SIMD_LANES))
        {
            let values = f64x4::from([
                value_chunk[0],
                value_chunk[1],
                value_chunk[2],
                value_chunk[3],
            ]);
            let current_mins =
                f64x4::from([min_chunk[0], min_chunk[1], min_chunk[2], min_chunk[3]]);
            let current_maxs =
                f64x4::from([max_chunk[0], max_chunk[1], max_chunk[2], max_chunk[3]]);
            min_chunk.copy_from_slice(&<[f64; SIMD_LANES]>::from(current_mins.min(values)));
            max_chunk.copy_from_slice(&<[f64; SIMD_LANES]>::from(current_maxs.max(values)));
            col += SIMD_LANES;
        }
        while col < cols {
            let value = row_values[col];
            mins[col] = mins[col].min(value);
            maxs[col] = maxs[col].max(value);
            col += 1;
        }
    }

    (mins, maxs)
}

/// Computes multiple percentiles with linear interpolation from one sort per column.
pub(crate) fn column_percentiles(
    x: &Array,
    rows: usize,
    cols: usize,
    quantiles: &[f64],
) -> Vec<Vec<f64>> {
    let rank_scales = quantiles
        .iter()
        .map(|q| q.clamp(0.0, 1.0) * (rows.saturating_sub(1)) as f64)
        .collect::<Vec<_>>();
    let data = x.data();
    let mut results = (0..quantiles.len())
        .map(|_| Vec::with_capacity(cols))
        .collect::<Vec<_>>();

    for col in 0..cols {
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            values.push(data[row * cols + col]);
        }
        values.sort_by(f64::total_cmp);

        for (result, rank_scale) in results.iter_mut().zip(&rank_scales) {
            let lower_index = rank_scale.floor() as usize;
            let upper_index = rank_scale.ceil() as usize;
            let fraction = rank_scale - lower_index as f64;
            let lower = values[lower_index];
            let upper = values[upper_index];
            result.push(lower + (upper - lower) * fraction);
        }
    }

    results
}

/// Collects sorted unique values for a single feature column.
pub(crate) fn unique_sorted(x: &Array, rows: usize, cols: usize, col: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        values.push(x.data()[row * cols + col]);
    }
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| left.total_cmp(right).is_eq());
    values
}

/// Collects sorted unique values from a vector input.
pub(crate) fn unique_sorted_1d(y: &Array) -> Vec<f64> {
    let mut values = y.to_vec();
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| left.total_cmp(right).is_eq());
    values
}
