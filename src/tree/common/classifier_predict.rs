use rayon::prelude::*;
use wide::f64x4;

use crate::darray::Array;

use super::classifier_fit::{ClassifierNode, ClassifierTree};

const PAR_THRESHOLD: usize = 16_384;

pub(crate) fn tree_proba_matrix(tree: &ClassifierTree, x: &Array, n_classes: usize) -> Array {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut data = vec![0.0; rows * n_classes];

    let fill = |(row, output): (usize, &mut [f64])| {
        let input = &x.data()[row * cols..(row + 1) * cols];
        output.copy_from_slice(predict_node(&tree.root, input));
    };

    if rows * cols >= PAR_THRESHOLD {
        data.par_chunks_mut(n_classes).enumerate().for_each(fill);
    } else {
        data.chunks_mut(n_classes).enumerate().for_each(fill);
    }

    Array::from_shape_vec(&[rows, n_classes], data)
}

pub(crate) fn ensemble_proba(trees: &[ClassifierTree], x: &Array, n_classes: usize) -> Array {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut data = vec![0.0; rows * n_classes];

    let fill = |(row, output): (usize, &mut [f64])| {
        let input = &x.data()[row * cols..(row + 1) * cols];
        for tree in trees {
            add_probabilities(output, predict_node(&tree.root, input));
        }
        let scale = 1.0 / trees.len() as f64;
        output.iter_mut().for_each(|value| *value *= scale);
    };

    if rows * cols * trees.len() >= PAR_THRESHOLD {
        data.par_chunks_mut(n_classes).enumerate().for_each(fill);
    } else {
        data.chunks_mut(n_classes).enumerate().for_each(fill);
    }

    Array::from_shape_vec(&[rows, n_classes], data)
}

pub(crate) fn predicted_classes(proba: &Array, classes: &Array) -> Array {
    let cols = proba.shape()[1];
    let mut predicted = Vec::with_capacity(proba.shape()[0]);
    for row in 0..proba.shape()[0] {
        let offset = row * cols;
        let best = (0..cols)
            .max_by(|left, right| {
                proba.data()[offset + *left].total_cmp(&proba.data()[offset + *right])
            })
            .unwrap_or(0);
        predicted.push(classes.data()[best]);
    }
    Array::array(&predicted)
}

fn predict_node<'a>(node: &'a ClassifierNode, row: &[f64]) -> &'a [f64] {
    match node {
        ClassifierNode::Leaf(values) => values,
        ClassifierNode::Split {
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

fn add_probabilities(output: &mut [f64], values: &[f64]) {
    let simd_len = output.len() / 4 * 4;
    for offset in (0..simd_len).step_by(4) {
        let left = f64x4::from([
            output[offset],
            output[offset + 1],
            output[offset + 2],
            output[offset + 3],
        ]);
        let right = f64x4::from([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let result: [f64; 4] = (left + right).into();
        output[offset..offset + 4].copy_from_slice(&result);
    }

    for index in simd_len..output.len() {
        output[index] += values[index];
    }
}
