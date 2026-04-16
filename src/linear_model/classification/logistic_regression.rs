use crate::darray::Array;

use rayon::prelude::*;

use super::super::common::{
    LinearModelError, fit_logistic_parameters, prepare_class_labels, validate_features,
};

const PAR_THRESHOLD: usize = 16_384;

/// Fits multinomial logistic regression models with gradient descent.
#[derive(Debug, Clone, PartialEq)]
pub struct LogisticRegression {
    pub fit_intercept: bool,
    pub alpha: f64,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub tol: f64,
    pub classes_: Option<Array>,
    pub coef_: Option<Array>,
    pub intercept_: Option<Array>,
    pub n_features_in_: Option<usize>,
}

impl Default for LogisticRegression {
    /// Returns the default logistic-regression configuration.
    fn default() -> Self {
        Self {
            fit_intercept: true,
            alpha: 0.0,
            max_iter: 1_000,
            learning_rate: 0.1,
            tol: 1e-4,
            classes_: None,
            coef_: None,
            intercept_: None,
            n_features_in_: None,
        }
    }
}

impl LogisticRegression {
    /// Creates a logistic regression model with default settings.
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

    /// Returns a copy of the model with a specific learning rate.
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Returns a copy of the model with a specific convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Reports whether the model has learned classes, coefficients, and metadata.
    pub fn is_fitted(&self) -> bool {
        self.classes_.is_some()
            && self.coef_.is_some()
            && self.intercept_.is_some()
            && self.n_features_in_.is_some()
    }

    /// Returns the fitted class labels.
    pub fn classes(&self) -> Result<&Array, LinearModelError> {
        self.classes_.as_ref().ok_or(LinearModelError::NotFitted)
    }

    /// Returns the fitted coefficient matrix.
    pub fn coef(&self) -> Result<&Array, LinearModelError> {
        self.coef_.as_ref().ok_or(LinearModelError::NotFitted)
    }

    /// Returns the fitted intercept vector.
    pub fn intercept(&self) -> Result<&Array, LinearModelError> {
        self.intercept_.as_ref().ok_or(LinearModelError::NotFitted)
    }

    /// Fits the model to a feature matrix and label vector.
    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, LinearModelError> {
        validate_features(x)?;
        let encoded = prepare_class_labels(x, y)?;
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(LinearModelError::InvalidAlpha(self.alpha));
        }
        if self.max_iter == 0 {
            return Err(LinearModelError::InvalidMaxIterations(self.max_iter));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(LinearModelError::InvalidLearningRate(self.learning_rate));
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(LinearModelError::InvalidTolerance(self.tol));
        }

        let (coef, intercept) = fit_logistic_parameters(
            x,
            &encoded.encoded,
            encoded.classes.len(),
            self.fit_intercept,
            self.max_iter,
            self.learning_rate,
            self.alpha,
            self.tol,
        );

        self.classes_ = Some(Array::from_shape_vec(
            &[encoded.classes.len()],
            encoded.classes,
        ));
        self.coef_ = Some(coef.transpose());
        self.intercept_ = Some(intercept);
        self.n_features_in_ = Some(x.shape()[1]);
        Ok(self)
    }

    /// Returns class scores before softmax normalization.
    pub fn decision_function(&self, x: &Array) -> Result<Array, LinearModelError> {
        let classes = self.classes_.as_ref().ok_or(LinearModelError::NotFitted)?;
        let scores = logistic_scores(self, x)?;
        let rows = scores.shape()[0];
        let n_classes = classes.len();

        if n_classes == 2 {
            let margins = (0..rows)
                .map(|row| scores.data()[row * n_classes + 1] - scores.data()[row * n_classes])
                .collect::<Vec<_>>();
            Ok(Array::array(&margins))
        } else {
            Ok(scores)
        }
    }

    /// Predicts class probabilities for a feature matrix.
    pub fn predict_proba(&self, x: &Array) -> Result<Array, LinearModelError> {
        let scores = logistic_scores(self, x)?;
        let rows = scores.shape()[0];
        let n_classes = scores.shape()[1];
        let mut data = scores.to_vec();

        let normalize = |row: &mut [f64]| {
            let max_score = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut normalizer = 0.0;
            for value in row.iter_mut() {
                *value = (*value - max_score).exp();
                normalizer += *value;
            }
            for value in row.iter_mut() {
                *value /= normalizer;
            }
        };

        if rows * n_classes >= PAR_THRESHOLD {
            data.par_chunks_mut(n_classes).for_each(normalize);
        } else {
            data.chunks_mut(n_classes).for_each(normalize);
        }

        Ok(Array::from_shape_vec(&[rows, n_classes], data))
    }

    /// Predicts log-probabilities for a feature matrix.
    pub fn predict_log_proba(&self, x: &Array) -> Result<Array, LinearModelError> {
        Ok(self.predict_proba(x)?.log())
    }

    /// Predicts class labels for a feature matrix.
    pub fn predict(&self, x: &Array) -> Result<Array, LinearModelError> {
        let probabilities = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().ok_or(LinearModelError::NotFitted)?;
        let rows = probabilities.shape()[0];
        let n_classes = probabilities.shape()[1];
        let mut predicted = Vec::with_capacity(rows);

        for row in 0..rows {
            let offset = row * n_classes;
            let (best_class, _) = (0..n_classes)
                .map(|class| (class, probabilities.data()[offset + class]))
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .ok_or(LinearModelError::EmptyInput)?;
            predicted.push(classes.data()[best_class]);
        }

        Ok(Array::array(&predicted))
    }

    /// Returns the classification accuracy on the provided data.
    pub fn score(&self, x: &Array, y: &Array) -> Result<f64, LinearModelError> {
        let prediction = self.predict(x)?;
        crate::metrics::accuracy_score(y, &prediction)
            .map_err(|_| LinearModelError::InvalidLabelShape(y.shape().to_vec()))
    }
}

fn logistic_scores(model: &LogisticRegression, x: &Array) -> Result<Array, LinearModelError> {
    validate_features(x)?;
    let expected = model.n_features_in_.ok_or(LinearModelError::NotFitted)?;
    if x.shape()[1] != expected {
        return Err(LinearModelError::FeatureCountMismatch {
            expected,
            got: x.shape()[1],
        });
    }

    let coefficients = model.coef_.as_ref().ok_or(LinearModelError::NotFitted)?;
    let intercepts = model
        .intercept_
        .as_ref()
        .ok_or(LinearModelError::NotFitted)?;
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let n_classes = coefficients.shape()[0];
    let mut data = vec![0.0; rows * n_classes];

    let compute = |(row, output): (usize, &mut [f64])| {
        let input = &x.data()[row * cols..(row + 1) * cols];
        for class in 0..n_classes {
            let mut score = intercepts.data()[class];
            let weights = &coefficients.data()[class * cols..(class + 1) * cols];
            score += input
                .iter()
                .zip(weights)
                .map(|(feature, weight)| feature * weight)
                .sum::<f64>();
            output[class] = score;
        }
    };

    if rows * cols * n_classes >= PAR_THRESHOLD {
        data.par_chunks_mut(n_classes).enumerate().for_each(compute);
    } else {
        data.chunks_mut(n_classes).enumerate().for_each(compute);
    }

    Ok(Array::from_shape_vec(&[rows, n_classes], data))
}

impl crate::metrics::SupervisedEstimator for LogisticRegression {
    type Error = LinearModelError;

    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error> {
        LogisticRegression::fit(self, x, y)
    }

    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error> {
        LogisticRegression::score(self, x, y)
    }
}
