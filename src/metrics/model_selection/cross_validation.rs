use std::fmt::Display;

use crate::darray::{Array, RandomState};
use rayon::prelude::*;

use super::super::MetricsError;
use super::estimator::SupervisedEstimator;
use super::types::CrossValidationOptions;
use super::validation::validate_split_inputs;

/// Returns one validation score per fold for a supervised estimator.
pub fn cross_val_score<E>(
    estimator: &E,
    x: &Array,
    y: &Array,
    options: CrossValidationOptions,
) -> Result<Array, MetricsError>
where
    E: SupervisedEstimator + Send + Sync,
    E::Error: Display,
{
    let samples = validate_split_inputs(x, y)?;
    if options.cv < 2 {
        return Err(MetricsError::InvalidCv(options.cv));
    }
    if options.cv > samples {
        return Err(MetricsError::InvalidSplitSize {
            name: "cv",
            details: "cannot exceed the number of samples",
        });
    }

    let mut indices = (0..samples).collect::<Vec<_>>();
    if options.shuffle {
        if let Some(seed) = options.random_state {
            shuffle_indices(&mut indices, seed);
        } else {
            shuffle_indices_unseeded(&mut indices);
        }
    }

    let fold_sizes = fold_sizes(samples, options.cv);
    let mut offset = 0;
    let folds = fold_sizes
        .into_iter()
        .map(|fold_size| {
            let range = (offset, offset + fold_size);
            offset += fold_size;
            range
        })
        .collect::<Vec<_>>();

    let fold_scores = folds
        .into_par_iter()
        .map(|(start, end)| {
            let test_indices = &indices[start..end];
            let mut train_indices = Vec::with_capacity(samples - test_indices.len());
            train_indices.extend_from_slice(&indices[..start]);
            train_indices.extend_from_slice(&indices[end..]);

            let x_train = x.take(&train_indices, 0);
            let y_train = y.take(&train_indices, 0);
            let x_test = x.take(test_indices, 0);
            let y_test = y.take(test_indices, 0);

            let mut model = estimator.clone();
            model
                .fit(&x_train, &y_train)
                .map_err(|error| MetricsError::EstimatorError(error.to_string()))?;
            model
                .score(&x_test, &y_test)
                .map_err(|error| MetricsError::EstimatorError(error.to_string()))
        })
        .collect::<Vec<_>>();

    let scores = fold_scores.into_iter().collect::<Result<Vec<_>, _>>()?;

    Ok(Array::array(&scores))
}

fn fold_sizes(samples: usize, cv: usize) -> Vec<usize> {
    let base = samples / cv;
    let remainder = samples % cv;
    (0..cv)
        .map(|fold| if fold < remainder { base + 1 } else { base })
        .collect()
}

fn shuffle_indices(indices: &mut [usize], seed: u64) {
    let mut rng = RandomState::seeded(seed);
    shuffle_indices_with_rng(indices, &mut rng);
}

fn shuffle_indices_unseeded(indices: &mut [usize]) {
    let mut rng = RandomState::new();
    shuffle_indices_with_rng(indices, &mut rng);
}

fn shuffle_indices_with_rng(indices: &mut [usize], rng: &mut RandomState) {
    rng.shuffle_indices(indices);
}
