use crate::darray::Array;
use crate::metrics;

use super::lasso::Lasso;
use super::linear_regression::LinearRegression;
use super::prediction::predict_from_parameters;
use super::LinearModelError;
use super::Ridge;

impl LinearRegression {
    /// Predicts targets for a feature matrix.
    pub fn predict(&self, x: &Array) -> Result<Array, LinearModelError> {
        let coefficients = self.coef_.as_ref().ok_or(LinearModelError::NotFitted)?;
        let intercepts = self
            .intercept_
            .as_ref()
            .ok_or(LinearModelError::NotFitted)?;
        let expected = self.n_features_in_.ok_or(LinearModelError::NotFitted)?;
        predict_from_parameters(x, coefficients, intercepts, expected)
    }

    /// Returns the coefficient of determination on the provided data.
    pub fn score(&self, x: &Array, y: &Array) -> Result<f64, LinearModelError> {
        let prediction = self.predict(x)?;
        metrics::r2_score(y, &prediction)
            .map_err(|_| LinearModelError::InvalidTargetShape(y.shape().to_vec()))
    }
}

impl Ridge {
    /// Predicts targets for a feature matrix.
    pub fn predict(&self, x: &Array) -> Result<Array, LinearModelError> {
        let coefficients = self.coef_.as_ref().ok_or(LinearModelError::NotFitted)?;
        let intercepts = self
            .intercept_
            .as_ref()
            .ok_or(LinearModelError::NotFitted)?;
        let expected = self.n_features_in_.ok_or(LinearModelError::NotFitted)?;
        predict_from_parameters(x, coefficients, intercepts, expected)
    }

    /// Returns the coefficient of determination on the provided data.
    pub fn score(&self, x: &Array, y: &Array) -> Result<f64, LinearModelError> {
        let prediction = self.predict(x)?;
        metrics::r2_score(y, &prediction)
            .map_err(|_| LinearModelError::InvalidTargetShape(y.shape().to_vec()))
    }
}

impl Lasso {
    /// Predicts targets for a feature matrix.
    pub fn predict(&self, x: &Array) -> Result<Array, LinearModelError> {
        let coefficients = self.coef_.as_ref().ok_or(LinearModelError::NotFitted)?;
        let intercepts = self
            .intercept_
            .as_ref()
            .ok_or(LinearModelError::NotFitted)?;
        let expected = self.n_features_in_.ok_or(LinearModelError::NotFitted)?;
        predict_from_parameters(x, coefficients, intercepts, expected)
    }

    /// Returns the coefficient of determination on the provided data.
    pub fn score(&self, x: &Array, y: &Array) -> Result<f64, LinearModelError> {
        let prediction = self.predict(x)?;
        metrics::r2_score(y, &prediction)
            .map_err(|_| LinearModelError::InvalidTargetShape(y.shape().to_vec()))
    }
}
