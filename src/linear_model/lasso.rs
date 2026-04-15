use crate::darray::Array;

use super::LinearModelError;

/// Fits L1-regularized linear regression models with coordinate descent.
#[derive(Debug, Clone, PartialEq)]
pub struct Lasso {
    pub fit_intercept: bool,
    pub alpha: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub coef_: Option<Array>,
    pub intercept_: Option<Array>,
    pub n_features_in_: Option<usize>,
}

impl Lasso {
    /// Creates a lasso regression model with a specific regularization strength.
    pub fn new(alpha: f64) -> Self {
        Self {
            fit_intercept: true,
            alpha,
            max_iter: 1_000,
            tol: 1e-4,
            coef_: None,
            intercept_: None,
            n_features_in_: None,
        }
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

    /// Returns a copy of the model with a specific regularization strength.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Returns a copy of the model with a specific iteration cap.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Returns a copy of the model with a specific convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
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
