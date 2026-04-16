use fastrand::Rng;

use crate::darray::Array;

use super::{
    ClassifierParams, best_random_split, best_split, class_counts, partition_rows, probabilities,
    resolved_max_features, sample_features,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ClassifierNode {
    Leaf(Vec<f64>),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<ClassifierNode>,
        right: Box<ClassifierNode>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClassifierTree {
    pub(crate) root: ClassifierNode,
    pub(crate) feature_importances: Vec<f64>,
}

pub(crate) fn build_classifier_tree(
    x: &Array,
    y: &[usize],
    n_classes: usize,
    params: ClassifierParams,
    seed: u64,
) -> ClassifierTree {
    let mut rng = Rng::with_seed(seed);
    let mut importances = vec![0.0; x.shape()[1]];
    let rows = (0..x.shape()[0]).collect::<Vec<_>>();
    let root = build_node(
        x,
        y,
        &rows,
        n_classes,
        params,
        0,
        &mut rng,
        &mut importances,
    );
    ClassifierTree {
        root,
        feature_importances: importances,
    }
}

fn build_node(
    x: &Array,
    y: &[usize],
    rows: &[usize],
    n_classes: usize,
    params: ClassifierParams,
    depth: usize,
    rng: &mut Rng,
    importances: &mut [f64],
) -> ClassifierNode {
    let counts = class_counts(y, rows, n_classes);
    if should_stop(&counts, rows.len(), params, depth) {
        return ClassifierNode::Leaf(probabilities(&counts, rows.len()));
    }

    let feature_count = resolved_max_features(params.max_features, x.shape()[1]);
    let features = sample_features(x.shape()[1], feature_count, rng);
    let best = if params.random_split {
        best_random_split(x, y, rows, &features, &counts, params, rng)
    } else {
        best_split(x, y, rows, &features, &counts, params)
    };
    let Some(best) = best else {
        return ClassifierNode::Leaf(probabilities(&counts, rows.len()));
    };

    let (left_rows, right_rows) = partition_rows(x, rows, best.feature, best.threshold);
    if left_rows.is_empty() || right_rows.is_empty() {
        return ClassifierNode::Leaf(probabilities(&counts, rows.len()));
    }

    importances[best.feature] += best.gain.max(0.0) * rows.len() as f64;
    let left = build_node(
        x,
        y,
        &left_rows,
        n_classes,
        params,
        depth + 1,
        rng,
        importances,
    );
    let right = build_node(
        x,
        y,
        &right_rows,
        n_classes,
        params,
        depth + 1,
        rng,
        importances,
    );

    ClassifierNode::Split {
        feature: best.feature,
        threshold: best.threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn should_stop(counts: &[usize], len: usize, params: ClassifierParams, depth: usize) -> bool {
    len < params.min_samples_split
        || counts.iter().filter(|count| **count != 0).count() <= 1
        || params.max_depth.is_some_and(|limit| depth >= limit)
}
