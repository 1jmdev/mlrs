use crate::darray::Array;

use super::LinearModelError;

/// Fits ordinary least-squares linear regression models.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegression {
    /// Controls whether the model centers the data and learns an intercept.
    pub fit_intercept: bool,
    /// Sets the number of gradient descent passes over the training data.
    pub epochs: usize,
    /// Sets the gradient descent step size.
    pub learning_rate: f64,
    /// Stores learned coefficients after fitting.
    pub coef_: Option<Array>,
    /// Stores learned intercept terms after fitting.
    pub intercept_: Option<Array>,
    /// Stores the feature count observed during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for LinearRegression {
    /// Returns the default linear regression configuration.
    fn default() -> Self {
        Self {
            fit_intercept: true,
            epochs: 1_000,
            learning_rate: 0.01,
            coef_: None,
            intercept_: None,
            n_features_in_: None,
        }
    }
}

impl LinearRegression {
    /// Creates a linear regression model with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of the model with a specific intercept setting.
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Returns a copy of the model with a specific intercept setting.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Returns a copy of the model with a specific epoch count.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Returns a copy of the model with a specific learning rate.
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Reports whether the model has learned coefficients and metadata.
    pub fn is_fitted(&self) -> bool {
        self.coef_.is_some() && self.intercept_.is_some() && self.n_features_in_.is_some()
    }

    /// Returns the fitted coefficient array.
    pub fn coef(&self) -> Result<&Array, LinearModelError> {
        self.coef_.as_ref().ok_or(LinearModelError::NotFitted)
    }

    /// Returns the fitted intercept array.
    pub fn intercept(&self) -> Result<&Array, LinearModelError> {
        self.intercept_.as_ref().ok_or(LinearModelError::NotFitted)
    }
}
