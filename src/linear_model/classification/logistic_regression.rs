use crate::darray::Array;

use rayon::prelude::*;
use wide::f64x4;

use super::super::common::{
    LinearModelError, dot_simd, scaled_add_assign, validate_features,
};

const SIMD_WIDTH: usize = 4;
const PAR_THRESHOLD: usize = 16_384;

#[derive(Debug, Clone, PartialEq)]
struct PreparedClasses {
    classes: Vec<f64>,
    encoded: Vec<usize>,
}

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
            score += dot_simd(input, weights);
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

fn prepare_class_labels(x: &Array, y: &Array) -> Result<PreparedClasses, LinearModelError> {
    if !y.is_vector() || y.is_empty() {
        return Err(LinearModelError::InvalidLabelShape(y.shape().to_vec()));
    }
    if y.len() != x.shape()[0] {
        return Err(LinearModelError::SampleCountMismatch {
            x_samples: x.shape()[0],
            y_samples: y.len(),
        });
    }
    if y.data().iter().any(|value| !value.is_finite()) {
        return Err(LinearModelError::InvalidLabelShape(y.shape().to_vec()));
    }

    let mut classes = y.to_vec();
    classes.sort_by(f64::total_cmp);
    classes.dedup_by(|left, right| left.total_cmp(right).is_eq());
    if classes.len() < 2 {
        return Err(LinearModelError::InvalidClassCount(classes.len()));
    }

    let encoded = y
        .data()
        .iter()
        .map(|value| {
            classes
                .binary_search_by(|class| class.total_cmp(value))
                .map_err(|_| LinearModelError::InvalidLabelShape(y.shape().to_vec()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PreparedClasses { classes, encoded })
}

fn fit_logistic_parameters(
    x: &Array,
    encoded_y: &[usize],
    n_classes: usize,
    fit_intercept: bool,
    max_iter: usize,
    learning_rate: f64,
    alpha: f64,
    tol: f64,
) -> (Array, Array) {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let mut coefficients = vec![0.0; n_features * n_classes];
    let mut intercepts = vec![0.0; n_classes];
    let mut probabilities = vec![0.0; n_samples * n_classes];
    let scale = 1.0 / n_samples as f64;

    for _ in 0..max_iter {
        compute_logistic_probabilities(
            x,
            &coefficients,
            &intercepts,
            fit_intercept,
            n_classes,
            &mut probabilities,
        );

        let (gradient_w, gradient_b) =
            logistic_gradients(x, encoded_y, n_classes, &probabilities, n_features);

        let mut max_update = apply_logistic_gradient(
            &mut coefficients,
            &gradient_w,
            learning_rate * scale,
            learning_rate * alpha,
        );
        if fit_intercept {
            for class in 0..n_classes {
                let update = learning_rate * gradient_b[class] * scale;
                intercepts[class] -= update;
                max_update = max_update.max(update.abs());
            }
        }
        if max_update <= tol {
            break;
        }
    }

    (
        Array::from_shape_vec(&[n_features, n_classes], coefficients),
        Array::from_shape_vec(&[n_classes], intercepts),
    )
}

fn compute_logistic_probabilities(
    x: &Array,
    coefficients: &[f64],
    intercepts: &[f64],
    fit_intercept: bool,
    n_classes: usize,
    probabilities: &mut [f64],
) {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let compute_row = |(sample, row): (usize, &mut [f64])| {
        let x_offset = sample * n_features;
        let mut max_score = f64::NEG_INFINITY;
        for class in 0..n_classes {
            let mut score = if fit_intercept { intercepts[class] } else { 0.0 };
            for feature in 0..n_features {
                score += x.data()[x_offset + feature] * coefficients[feature * n_classes + class];
            }
            row[class] = score;
            max_score = max_score.max(score);
        }

        let mut normalizer = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max_score).exp();
            normalizer += *value;
        }
        for value in row.iter_mut() {
            *value /= normalizer;
        }
    };

    if n_samples * n_features * n_classes >= PAR_THRESHOLD {
        probabilities
            .par_chunks_mut(n_classes)
            .enumerate()
            .for_each(compute_row);
    } else {
        probabilities
            .chunks_mut(n_classes)
            .enumerate()
            .for_each(compute_row);
    }
}

fn logistic_gradients(
    x: &Array,
    encoded_y: &[usize],
    n_classes: usize,
    probabilities: &[f64],
    n_features: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_samples = x.shape()[0];
    if n_samples * n_features * n_classes >= PAR_THRESHOLD {
        (0..n_samples)
            .into_par_iter()
            .fold(
                || (vec![0.0; n_features * n_classes], vec![0.0; n_classes]),
                |(mut local_w, mut local_b), sample| {
                    accumulate_logistic_gradient(
                        &mut local_w,
                        &mut local_b,
                        &x.data()[sample * n_features..(sample + 1) * n_features],
                        &probabilities[sample * n_classes..(sample + 1) * n_classes],
                        encoded_y[sample],
                        n_classes,
                    );
                    (local_w, local_b)
                },
            )
            .reduce(
                || (vec![0.0; n_features * n_classes], vec![0.0; n_classes]),
                |mut left, right| {
                    scaled_add_assign(&mut left.0, &right.0, 1.0);
                    for (left_value, right_value) in left.1.iter_mut().zip(right.1) {
                        *left_value += right_value;
                    }
                    left
                },
            )
    } else {
        let mut gradient_w = vec![0.0; n_features * n_classes];
        let mut gradient_b = vec![0.0; n_classes];
        for sample in 0..n_samples {
            accumulate_logistic_gradient(
                &mut gradient_w,
                &mut gradient_b,
                &x.data()[sample * n_features..(sample + 1) * n_features],
                &probabilities[sample * n_classes..(sample + 1) * n_classes],
                encoded_y[sample],
                n_classes,
            );
        }
        (gradient_w, gradient_b)
    }
}

fn accumulate_logistic_gradient(
    gradient_w: &mut [f64],
    gradient_b: &mut [f64],
    sample_features: &[f64],
    sample_probabilities: &[f64],
    encoded_y: usize,
    n_classes: usize,
) {
    for class in 0..n_classes {
        let target = if encoded_y == class { 1.0 } else { 0.0 };
        let residual = sample_probabilities[class] - target;
        gradient_b[class] += residual;
        for (feature, value) in sample_features.iter().enumerate() {
            gradient_w[feature * n_classes + class] += value * residual;
        }
    }
}

fn apply_logistic_gradient(
    coefficients: &mut [f64],
    gradient: &[f64],
    gradient_scale: f64,
    l2_scale: f64,
) -> f64 {
    let gradient_scale_values = f64x4::splat(gradient_scale);
    let l2_scale_values = f64x4::splat(1.0 - l2_scale);
    let simd_len = coefficients.len() / SIMD_WIDTH * SIMD_WIDTH;
    let mut max_update = 0.0_f64;

    for offset in (0..simd_len).step_by(SIMD_WIDTH) {
        let original = f64x4::from([
            coefficients[offset],
            coefficients[offset + 1],
            coefficients[offset + 2],
            coefficients[offset + 3],
        ]);
        let updated = original * l2_scale_values
            - f64x4::from([
                gradient[offset],
                gradient[offset + 1],
                gradient[offset + 2],
                gradient[offset + 3],
            ]) * gradient_scale_values;
        let original_values: [f64; SIMD_WIDTH] = original.into();
        let updated_values: [f64; SIMD_WIDTH] = updated.into();
        coefficients[offset..offset + SIMD_WIDTH].copy_from_slice(&updated_values);
        for lane in 0..SIMD_WIDTH {
            max_update = max_update.max((updated_values[lane] - original_values[lane]).abs());
        }
    }
    for index in simd_len..coefficients.len() {
        let original = coefficients[index];
        coefficients[index] = original * (1.0 - l2_scale) - gradient[index] * gradient_scale;
        max_update = max_update.max((coefficients[index] - original).abs());
    }

    max_update
}
