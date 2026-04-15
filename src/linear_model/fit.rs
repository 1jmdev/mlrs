use crate::darray::Array;

use super::lasso::Lasso;
use super::linear_regression::LinearRegression;
use super::training::{
    finalize_parameters, fit_lasso_coefficients, fit_linear_coefficients, fit_ridge_coefficients,
    prepare_training_data,
};
use super::LinearModelError;
use super::Ridge;

impl LinearRegression {
    /// Fits the model to a feature matrix and target array.
    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, LinearModelError> {
        if self.epochs == 0 {
            return Err(LinearModelError::InvalidEpochs(self.epochs));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(LinearModelError::InvalidLearningRate(self.learning_rate));
        }
        let training = prepare_training_data(x, y, self.fit_intercept)?;
        // Prefer the direct normal-equation solve for release-speed parity with
        // Ridge, but keep the gradient-descent path as a fallback for singular
        // design matrices.
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
            super::validation::prepare_targets(x, y)?.matrix
        };
        Ok(expected.sub_array(&prediction))
    }
}

impl Ridge {
    /// Fits the model to a feature matrix and target array.
    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, LinearModelError> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(LinearModelError::InvalidAlpha(self.alpha));
        }
        let training = prepare_training_data(x, y, self.fit_intercept)?;
        let coefficients = fit_ridge_coefficients(&training.x, &training.y, self.alpha)?;
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
}

impl Lasso {
    /// Fits the model to a feature matrix and target array.
    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, LinearModelError> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(LinearModelError::InvalidAlpha(self.alpha));
        }
        if self.max_iter == 0 {
            return Err(LinearModelError::InvalidMaxIterations(self.max_iter));
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(LinearModelError::InvalidTolerance(self.tol));
        }

        let training = prepare_training_data(x, y, self.fit_intercept)?;
        let coefficients = fit_lasso_coefficients(
            &training.x,
            &training.y,
            self.alpha,
            self.max_iter,
            self.tol,
        );
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
}
