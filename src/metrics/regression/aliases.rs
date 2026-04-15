use crate::darray::Array;

use super::super::MetricsError;
use super::{mean_absolute_error, mean_squared_error, root_mean_squared_error};

/// Returns the mean absolute error using the common sklearn-style alias.
pub fn mae(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_absolute_error(y_true, y_pred)
}

/// Returns the mean squared error using the common sklearn-style alias.
pub fn mse(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    mean_squared_error(y_true, y_pred)
}

/// Returns the root mean squared error using the common sklearn-style alias.
pub fn rmse(y_true: &Array, y_pred: &Array) -> Result<f64, MetricsError> {
    root_mean_squared_error(y_true, y_pred)
}
