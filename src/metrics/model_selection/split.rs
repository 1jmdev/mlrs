use crate::darray::{Array, RandomState};

use super::super::MetricsError;
use super::types::{SplitData, TrainTestSplitOptions};
use super::validation::{resolve_split_sizes, validate_split_inputs};

/// Splits feature and target arrays into train and test subsets.
pub fn train_test_split(
    x: &Array,
    y: &Array,
    options: TrainTestSplitOptions,
) -> Result<SplitData, MetricsError> {
    let samples = validate_split_inputs(x, y)?;
    let (train_size, test_size) =
        resolve_split_sizes(samples, options.train_size, options.test_size)?;
    let mut indices = (0..samples).collect::<Vec<_>>();

    if options.shuffle {
        if let Some(seed) = options.random_state {
            RandomState::seeded(seed).shuffle_indices(&mut indices);
        } else {
            RandomState::new().shuffle_indices(&mut indices);
        }
    }

    let train_indices = &indices[..train_size];
    let test_indices = &indices[train_size..train_size + test_size];

    Ok(SplitData {
        x_train: x.take(train_indices, 0),
        x_test: x.take(test_indices, 0),
        y_train: y.take(train_indices, 0),
        y_test: y.take(test_indices, 0),
    })
}
