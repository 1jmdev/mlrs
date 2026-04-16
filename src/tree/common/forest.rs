use crate::darray::Array;

use super::{ClassifierTree, TreeError};

pub(crate) fn validate_ensemble_params(
    n_estimators: usize,
    max_depth: Option<usize>,
    min_split: usize,
    min_leaf: usize,
) -> Result<(), TreeError> {
    if n_estimators == 0 {
        return Err(TreeError::InvalidEstimatorCount(0));
    }
    if max_depth == Some(0) {
        return Err(TreeError::InvalidMaxDepth(0));
    }
    if min_split < 2 {
        return Err(TreeError::InvalidMinSamplesSplit(min_split));
    }
    if min_leaf == 0 {
        return Err(TreeError::InvalidMinSamplesLeaf(min_leaf));
    }
    Ok(())
}

pub(crate) fn sampled_dataset(
    x: &Array,
    y: &[usize],
    bootstrap: bool,
    seed: u64,
) -> (Array, Vec<usize>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let indices = if bootstrap {
        (0..rows).map(|_| rng.usize(0..rows)).collect::<Vec<_>>()
    } else {
        (0..rows).collect::<Vec<_>>()
    };
    let mut data = Vec::with_capacity(indices.len() * cols);
    let mut labels = Vec::with_capacity(indices.len());
    for index in indices {
        data.extend_from_slice(&x.data()[index * cols..(index + 1) * cols]);
        labels.push(y[index]);
    }
    (Array::from_shape_vec(&[labels.len(), cols], data), labels)
}

pub(crate) fn average_importances(trees: &[ClassifierTree], n_features: usize) -> Array {
    let mut values = vec![0.0; n_features];
    for tree in trees {
        for (index, value) in tree.feature_importances.iter().enumerate() {
            values[index] += *value;
        }
    }
    let total = values.iter().sum::<f64>();
    if total > 0.0 {
        values.iter_mut().for_each(|value| *value /= total);
    }
    Array::array(&values)
}
