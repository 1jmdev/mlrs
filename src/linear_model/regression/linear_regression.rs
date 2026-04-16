use crate::darray::Array;

use super::super::common::{
    LinearModelError, finalize_parameters, fit_linear_coefficients, fit_ridge_coefficients,
    predict_from_parameters, prepare_targets, prepare_training_data,
};

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

    /// Fits the model to a feature matrix and target array.
    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, LinearModelError> {
        if self.epochs == 0 {
            return Err(LinearModelError::InvalidEpochs(self.epochs));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(LinearModelError::InvalidLearningRate(self.learning_rate));
        }
        let training = prepare_training_data(x, y, self.fit_intercept)?;
        let coefficients =
            fit_ridge_coefficients(&training.x, &training.y, 0.0).unwrap_or_else(|_| {
                fit_linear_coefficients(&training.x, &training.y, self.epochs, self.learning_rate)
            });
        let (coef, intercept) = finalize_parameters(
            &coefficients,
            &training.prepared_y,
            &training.x_offset,
            &training.y_offset,
            self.fit_intercept,
        );

        self.n_features_in_ = Some(training.n_features);
        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);

        Ok(self)
    }

    /// Returns observed targets minus model predictions.
    pub fn residuals(&self, x: &Array, y: &Array) -> Result<Array, LinearModelError> {
        let prediction = self.predict(x)?;
        let expected = if prediction.ndim() == 1 {
            y.copy()
        } else {
            prepare_targets(x, y)?.matrix
        };
        Ok(expected.sub_array(&prediction))
    }

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
        crate::metrics::r2_score(y, &prediction)
            .map_err(|_| LinearModelError::InvalidTargetShape(y.shape().to_vec()))
    }
}

impl crate::metrics::SupervisedEstimator for LinearRegression {
    type Error = LinearModelError;

    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error> {
        LinearRegression::fit(self, x, y)
    }

    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error> {
        LinearRegression::score(self, x, y)
    }
}
