use crate::darray::Array;

use super::LinearModelError;

/// Fits L2-regularized linear regression models.
#[derive(Debug, Clone, PartialEq)]
pub struct Ridge {
    pub fit_intercept: bool,
    pub alpha: f64,
    pub coef_: Option<Array>,
    pub intercept_: Option<Array>,
    pub n_features_in_: Option<usize>,
}

impl Ridge {
    /// Creates a ridge regression model with a specific regularization strength.
    pub fn new(alpha: f64) -> Self {
        Self {
            fit_intercept: true,
            alpha,
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
