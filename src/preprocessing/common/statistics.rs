use crate::darray::Array;

/// Computes per-feature means and population variances.
pub(crate) fn column_mean_var(x: &Array, rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0; cols];
    for row in 0..rows {
        let offset = row * cols;
        for (col, mean) in means.iter_mut().enumerate() {
            *mean += x.data()[offset + col];
        }
    }
    for mean in &mut means {
        *mean /= rows as f64;
    }

    let mut variances = vec![0.0; cols];
    for row in 0..rows {
        let offset = row * cols;
        for col in 0..cols {
            let centered = x.data()[offset + col] - means[col];
            variances[col] += centered * centered;
        }
    }
    for variance in &mut variances {
        *variance /= rows as f64;
    }

    (means, variances)
}

/// Computes per-feature minima and maxima.
pub(crate) fn column_min_max(x: &Array, rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mins = vec![f64::INFINITY; cols];
    let mut maxs = vec![f64::NEG_INFINITY; cols];

    for row in 0..rows {
        let offset = row * cols;
        for col in 0..cols {
            let value = x.data()[offset + col];
            mins[col] = mins[col].min(value);
            maxs[col] = maxs[col].max(value);
        }
    }

    (mins, maxs)
}

/// Computes a percentile with linear interpolation between neighboring ranks.
pub(crate) fn column_percentile(x: &Array, rows: usize, cols: usize, q: f64) -> Vec<f64> {
    let rank_scale = q.clamp(0.0, 1.0) * (rows.saturating_sub(1)) as f64;
    let mut percentiles = Vec::with_capacity(cols);

    for col in 0..cols {
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            values.push(x.data()[row * cols + col]);
        }
        values.sort_by(f64::total_cmp);

        let lower_index = rank_scale.floor() as usize;
        let upper_index = rank_scale.ceil() as usize;
        let fraction = rank_scale - lower_index as f64;
        let lower = values[lower_index];
        let upper = values[upper_index];
        percentiles.push(lower + (upper - lower) * fraction);
    }

    percentiles
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
