use rayon::prelude::*;

use crate::darray::Array;

use super::{RegressionTree, predict_regression_tree};

const PAR_THRESHOLD: usize = 16_384;

pub(crate) fn class_priors(y: &[usize], n_classes: usize, rows: usize) -> Vec<f64> {
    let mut counts = vec![0.0; n_classes];
    for value in y {
        counts[*value] += 1.0;
    }
    counts
        .into_iter()
        .map(|count| (count / rows as f64).max(1e-12).ln())
        .collect()
}

pub(crate) fn sampled_rows(rows: usize, subsample: f64, seed: u64) -> Vec<usize> {
    let size = ((rows as f64) * subsample).round().max(1.0) as usize;
    let mut indices = (0..rows).collect::<Vec<_>>();
    let mut rng = fastrand::Rng::with_seed(seed);
    for index in 0..size.min(rows) {
        let swap = rng.usize(index..rows);
        indices.swap(index, swap);
    }
    indices.truncate(size.min(rows));
    indices
}

pub(crate) fn update_logits(
    logits: &mut [f64],
    x: &Array,
    cols: usize,
    n_classes: usize,
    rate: f64,
    trees: &[RegressionTree],
) {
    let update = |(row, output): (usize, &mut [f64])| {
        let input = &x.data()[row * cols..(row + 1) * cols];
        for class in 0..n_classes {
            output[class] += rate * predict_regression_tree(&trees[class], input);
        }
    };
    if x.shape()[0] * cols * n_classes >= PAR_THRESHOLD {
        logits
            .par_chunks_mut(n_classes)
            .enumerate()
            .for_each(update);
    } else {
        logits.chunks_mut(n_classes).enumerate().for_each(update);
    }
}

pub(crate) fn softmax_rows(logits: &[f64], rows: usize, n_classes: usize) -> Vec<f64> {
    let mut output = logits.to_vec();
    let normalize = |row: &mut [f64]| {
        let max_value = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        row.iter_mut().for_each(|value| {
            *value = (*value - max_value).exp();
            total += *value;
        });
        row.iter_mut().for_each(|value| *value /= total);
    };
    if rows * n_classes >= PAR_THRESHOLD {
        output.par_chunks_mut(n_classes).for_each(normalize);
    } else {
        output.chunks_mut(n_classes).for_each(normalize);
    }
    output
}
