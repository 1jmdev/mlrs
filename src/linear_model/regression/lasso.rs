use crate::darray::Array;

use rayon::prelude::*;

use super::super::common::{
    LinearModelError, compute_gram_and_xty, finalize_parameters, predict_from_parameters,
    prepare_training_data, scaled_add_assign, soft_threshold,
};

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

impl crate::metrics::SupervisedEstimator for Lasso {
    type Error = LinearModelError;

    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error> {
        Lasso::fit(self, x, y)
    }

    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error> {
        Lasso::score(self, x, y)
    }
}

fn fit_lasso_coefficients(x: &Array, y: &Array, alpha: f64, max_iter: usize, tol: f64) -> Array {
    let n_features = x.shape()[1];
    let n_targets = y.shape()[1];
    let n_samples = x.shape()[0];
    let (gram, xty) = compute_gram_and_xty(x, y);

    let feature_norms = (0..n_features)
        .map(|feature| gram[feature * n_features + feature])
        .collect::<Vec<_>>();
    let coefficients = (0..n_targets)
        .into_par_iter()
        .map(|target| {
            let target_xty = (0..n_features)
                .map(|feature| xty[feature * n_targets + target])
                .collect::<Vec<_>>();
            fit_lasso_target(
                &gram,
                &target_xty,
                &feature_norms,
                alpha,
                max_iter,
                tol,
                n_features,
                n_samples,
            )
        })
        .collect::<Vec<_>>();

    let mut data = vec![0.0; n_features * n_targets];
    for (target, values) in coefficients.into_iter().enumerate() {
        for (feature, value) in values.into_iter().enumerate() {
            data[feature * n_targets + target] = value;
        }
    }
    Array::from_shape_vec(&[n_features, n_targets], data)
}

fn fit_lasso_target(
    gram: &[f64],
    xty: &[f64],
    feature_norms: &[f64],
    alpha: f64,
    max_iter: usize,
    tol: f64,
    n_features: usize,
    n_samples: usize,
) -> Vec<f64> {
    let mut coefficients = vec![0.0; n_features];
    let mut correlations = vec![0.0; n_features];
    let penalty = alpha * n_samples as f64;

    for _ in 0..max_iter {
        let mut max_update = 0.0_f64;
        for feature in 0..n_features {
            let norm = feature_norms[feature];
            if norm <= f64::EPSILON {
                continue;
            }

            let old = coefficients[feature];
            let rho = xty[feature] - (correlations[feature] - norm * old);
            let updated = soft_threshold(rho, penalty) / norm;
            let delta = updated - old;
            if delta != 0.0 {
                coefficients[feature] = updated;
                let row = &gram[feature * n_features..(feature + 1) * n_features];
                scaled_add_assign(&mut correlations, row, delta);
                max_update = max_update.max(delta.abs());
            }
        }
        if max_update <= tol {
            break;
        }
    }

    coefficients
}
