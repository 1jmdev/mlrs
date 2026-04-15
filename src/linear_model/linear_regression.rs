use crate::darray::Array;

use super::LinearModelError;

#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegression {
    pub fit_intercept: bool,
    pub epochs: usize,
    pub learning_rate: f64,
    pub coef_: Option<Array>,
    pub intercept_: Option<Array>,
    pub n_features_in_: Option<usize>,
}

impl Default for LinearRegression {
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn is_fitted(&self) -> bool {
        self.coef_.is_some() && self.intercept_.is_some() && self.n_features_in_.is_some()
    }

    pub fn coef(&self) -> Result<&Array, LinearModelError> {
        self.coef_.as_ref().ok_or(LinearModelError::NotFitted)
    }

    pub fn intercept(&self) -> Result<&Array, LinearModelError> {
        self.intercept_.as_ref().ok_or(LinearModelError::NotFitted)
    }
}
