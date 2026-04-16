use fastrand::Rng;

use crate::darray::Array;

use super::{GradientBoostingParams, MaxFeatures, resolved_max_features};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RegressionNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<RegressionNode>,
        right: Box<RegressionNode>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionTree {
    pub(crate) root: RegressionNode,
}

pub(crate) fn build_regression_tree(
    x: &Array,
    target: &[f64],
    params: GradientBoostingParams,
    seed: u64,
    rows: &[usize],
) -> RegressionTree {
    let mut rng = Rng::with_seed(seed);
    let root = build_node(x, target, rows, params, 0, &mut rng);
    RegressionTree { root }
}

pub(crate) fn predict_regression_tree(tree: &RegressionTree, row: &[f64]) -> f64 {
    predict_node(&tree.root, row)
}

fn build_node(
    x: &Array,
    target: &[f64],
    rows: &[usize],
    params: GradientBoostingParams,
    depth: usize,
    rng: &mut Rng,
) -> RegressionNode {
    let mean = rows.iter().map(|row| target[*row]).sum::<f64>() / rows.len() as f64;
    if rows.len() < params.min_samples_split || depth >= params.max_depth {
        return RegressionNode::Leaf(mean);
    }

    let features = sample_features(x.shape()[1], params.max_features, rng);
    let Some((feature, threshold)) = best_split(x, target, rows, &features, params) else {
        return RegressionNode::Leaf(mean);
    };

    let (left_rows, right_rows) = partition_rows(x, rows, feature, threshold);
    if left_rows.len() < params.min_samples_leaf || right_rows.len() < params.min_samples_leaf {
        return RegressionNode::Leaf(mean);
    }

    RegressionNode::Split {
        feature,
        threshold,
        left: Box::new(build_node(x, target, &left_rows, params, depth + 1, rng)),
        right: Box::new(build_node(x, target, &right_rows, params, depth + 1, rng)),
    }
}

fn predict_node(node: &RegressionNode, row: &[f64]) -> f64 {
    match node {
        RegressionNode::Leaf(value) => *value,
        RegressionNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if row[*feature] <= *threshold {
                predict_node(left, row)
            } else {
                predict_node(right, row)
            }
        }
    }
}

fn sample_features(n_features: usize, max_features: MaxFeatures, rng: &mut Rng) -> Vec<usize> {
    let count = resolved_max_features(max_features, n_features);
    let mut features = (0..n_features).collect::<Vec<_>>();
    for index in 0..count.min(n_features) {
        let swap = rng.usize(index..n_features);
        features.swap(index, swap);
    }
    features.truncate(count.min(n_features));
    features
}

fn best_split(
    x: &Array,
    target: &[f64],
    rows: &[usize],
    features: &[usize],
    params: GradientBoostingParams,
) -> Option<(usize, f64)> {
    features
        .iter()
        .filter_map(|feature| best_feature_split(x, target, rows, *feature, params))
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(feature, _, threshold)| (feature, threshold))
}

fn best_feature_split(
    x: &Array,
    target: &[f64],
    rows: &[usize],
    feature: usize,
    params: GradientBoostingParams,
) -> Option<(usize, f64, f64)> {
    let cols = x.shape()[1];
    let mut pairs = rows
        .iter()
        .map(|row| (x.data()[row * cols + feature], target[*row]))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));

    let total_sum = pairs.iter().map(|pair| pair.1).sum::<f64>();
    let total_sq = pairs.iter().map(|pair| pair.1 * pair.1).sum::<f64>();
    let mut left_sum = 0.0;
    let mut left_sq = 0.0;
    let mut best = None;

    for index in 0..pairs.len() - 1 {
        left_sum += pairs[index].1;
        left_sq += pairs[index].1 * pairs[index].1;
        if pairs[index].0 == pairs[index + 1].0 {
            continue;
        }

        let left_len = index + 1;
        let right_len = pairs.len() - left_len;
        if left_len < params.min_samples_leaf || right_len < params.min_samples_leaf {
            continue;
        }

        let right_sum = total_sum - left_sum;
        let right_sq = total_sq - left_sq;
        let score = sse(left_len, left_sum, left_sq) + sse(right_len, right_sum, right_sq);
        let threshold = (pairs[index].0 + pairs[index + 1].0) * 0.5;
        if best
            .as_ref()
            .is_none_or(|current: &(usize, f64, f64)| score < current.1)
        {
            best = Some((feature, score, threshold));
        }
    }

    best
}

fn sse(len: usize, sum: f64, sum_sq: f64) -> f64 {
    if len == 0 {
        0.0
    } else {
        sum_sq - sum * sum / len as f64
    }
}

fn partition_rows(
    x: &Array,
    rows: &[usize],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let cols = x.shape()[1];
    let mut left = Vec::with_capacity(rows.len());
    let mut right = Vec::with_capacity(rows.len());
    for row in rows {
        if x.data()[row * cols + feature] <= threshold {
            left.push(*row);
        } else {
            right.push(*row);
        }
    }
    (left, right)
}
