use fastrand::Rng;

use crate::darray::Array;

use super::split::{SplitCandidate, impurity};
use super::{ClassifierParams, Criterion};

pub(crate) fn class_counts(y: &[usize], rows: &[usize], n_classes: usize) -> Vec<usize> {
    let mut counts = vec![0; n_classes];
    for row in rows {
        counts[y[*row]] += 1;
    }
    counts
}

pub(crate) fn probabilities(counts: &[usize], total: usize) -> Vec<f64> {
    counts
        .iter()
        .map(|count| *count as f64 / total as f64)
        .collect()
}

pub(crate) fn sample_features(n_features: usize, count: usize, rng: &mut Rng) -> Vec<usize> {
    let mut features = (0..n_features).collect::<Vec<_>>();
    for index in 0..count.min(n_features) {
        let swap = rng.usize(index..n_features);
        features.swap(index, swap);
    }
    features.truncate(count.min(n_features));
    features
}

pub(crate) fn partition_rows(
    x: &Array,
    rows: &[usize],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let cols = x.shape()[1];
    let mut left = Vec::with_capacity(rows.len());
    let mut right = Vec::with_capacity(rows.len());
    for row in rows {
        let value = x.data()[row * cols + feature];
        if value <= threshold {
            left.push(*row);
        } else {
            right.push(*row);
        }
    }
    (left, right)
}

pub(crate) fn best_split(
    x: &Array,
    y: &[usize],
    rows: &[usize],
    features: &[usize],
    counts: &[usize],
    params: ClassifierParams,
) -> Option<SplitCandidate> {
    features
        .iter()
        .filter_map(|feature| best_feature_split(x, y, rows, *feature, counts, params))
        .max_by(|left, right| left.gain.total_cmp(&right.gain))
}

pub(crate) fn best_random_split(
    x: &Array,
    y: &[usize],
    rows: &[usize],
    features: &[usize],
    counts: &[usize],
    params: ClassifierParams,
    rng: &mut Rng,
) -> Option<SplitCandidate> {
    let cols = x.shape()[1];
    let mut best = None;

    for feature in features {
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        for row in rows {
            let value = x.data()[row * cols + feature];
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }
        if min_value >= max_value {
            continue;
        }

        let threshold = rng.f64() * (max_value - min_value) + min_value;
        let gain = threshold_gain(
            x,
            y,
            rows,
            *feature,
            threshold,
            counts,
            params.criterion,
            params.min_samples_leaf,
        );
        let candidate = SplitCandidate {
            feature: *feature,
            threshold,
            gain,
        };
        if best
            .as_ref()
            .is_none_or(|current: &SplitCandidate| candidate.gain > current.gain)
        {
            best = Some(candidate);
        }
    }

    best.filter(|candidate| candidate.gain > 0.0)
}

fn best_feature_split(
    x: &Array,
    y: &[usize],
    rows: &[usize],
    feature: usize,
    counts: &[usize],
    params: ClassifierParams,
) -> Option<SplitCandidate> {
    let cols = x.shape()[1];
    let mut pairs = rows
        .iter()
        .map(|row| (x.data()[row * cols + feature], y[*row]))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut left = vec![0; counts.len()];
    let mut right = counts.to_vec();
    let parent = impurity(params.criterion, counts, pairs.len());
    let mut best = None;

    for index in 0..pairs.len() - 1 {
        let class = pairs[index].1;
        left[class] += 1;
        right[class] -= 1;
        if pairs[index].0 == pairs[index + 1].0 {
            continue;
        }

        let left_len = index + 1;
        let right_len = pairs.len() - left_len;
        if left_len < params.min_samples_leaf || right_len < params.min_samples_leaf {
            continue;
        }

        let gain = split_gain(
            parent,
            params.criterion,
            &left,
            &right,
            left_len,
            right_len,
            pairs.len(),
        );
        let candidate = SplitCandidate {
            feature,
            threshold: (pairs[index].0 + pairs[index + 1].0) * 0.5,
            gain,
        };
        if best
            .as_ref()
            .is_none_or(|current: &SplitCandidate| candidate.gain > current.gain)
        {
            best = Some(candidate);
        }
    }

    best.filter(|candidate| candidate.gain > 0.0)
}

fn threshold_gain(
    x: &Array,
    y: &[usize],
    rows: &[usize],
    feature: usize,
    threshold: f64,
    counts: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> f64 {
    let cols = x.shape()[1];
    let mut left = vec![0; counts.len()];
    let mut right = counts.to_vec();
    let mut left_len = 0;

    for row in rows {
        if x.data()[row * cols + feature] <= threshold {
            left[y[*row]] += 1;
            right[y[*row]] -= 1;
            left_len += 1;
        }
    }

    let right_len = rows.len() - left_len;
    if left_len < min_samples_leaf || right_len < min_samples_leaf {
        return f64::NEG_INFINITY;
    }

    split_gain(
        impurity(criterion, counts, rows.len()),
        criterion,
        &left,
        &right,
        left_len,
        right_len,
        rows.len(),
    )
}

fn split_gain(
    parent: f64,
    criterion: Criterion,
    left: &[usize],
    right: &[usize],
    left_len: usize,
    right_len: usize,
    total: usize,
) -> f64 {
    let left_score = impurity(criterion, left, left_len);
    let right_score = impurity(criterion, right, right_len);
    parent
        - (left_len as f64 / total as f64) * left_score
        - (right_len as f64 / total as f64) * right_score
}
