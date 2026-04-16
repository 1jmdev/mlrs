use rayon::prelude::*;

use crate::darray::Array;

use super::common::{
    GradientBoostingParams, MaxFeatures, RegressionTree, TreeError, build_regression_tree,
    class_priors, encode_labels, predicted_classes, sampled_rows, softmax_rows, update_logits,
    validate_fit_inputs, validate_predict_input,
};

#[derive(Debug, Clone, PartialEq)]
pub struct GradientBoostingMachine {
    pub learning_rate: f64,
    pub n_estimators: usize,
    pub subsample: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
    pub random_state: Option<u64>,
    pub classes_: Option<Array>,
    pub n_features_in_: Option<usize>,
    init_: Vec<f64>,
    stages: Vec<Vec<RegressionTree>>,
}

impl Default for GradientBoostingMachine {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            n_estimators: 100,
            subsample: 1.0,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            random_state: None,
            classes_: None,
            n_features_in_: None,
            init_: Vec::new(),
            stages: Vec::new(),
        }
    }
}

impl GradientBoostingMachine {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn learning_rate(mut self, value: f64) -> Self {
        self.learning_rate = value;
        self
    }
    pub fn n_estimators(mut self, value: usize) -> Self {
        self.n_estimators = value;
        self
    }
    pub fn subsample(mut self, value: f64) -> Self {
        self.subsample = value;
        self
    }
    pub fn max_depth(mut self, value: usize) -> Self {
        self.max_depth = value;
        self
    }
    pub fn min_samples_split(mut self, value: usize) -> Self {
        self.min_samples_split = value;
        self
    }
    pub fn min_samples_leaf(mut self, value: usize) -> Self {
        self.min_samples_leaf = value;
        self
    }
    pub fn max_features(mut self, value: MaxFeatures) -> Self {
        self.max_features = value;
        self
    }
    pub fn random_state(mut self, value: u64) -> Self {
        self.random_state = Some(value);
        self
    }
    pub fn is_fitted(&self) -> bool {
        !self.stages.is_empty() && self.classes_.is_some()
    }

    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, TreeError> {
        validate_fit_inputs(x, y)?;
        validate_params(
            self.learning_rate,
            self.n_estimators,
            self.subsample,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        )?;
        let encoded = encode_labels(y)?;
        let rows = x.shape()[0];
        let cols = x.shape()[1];
        let n_classes = encoded.classes.len();
        let params = GradientBoostingParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
        };

        self.init_ = class_priors(&encoded.encoded, n_classes, rows);
        self.stages.clear();
        let mut logits = vec![0.0; rows * n_classes];
        for row in 0..rows {
            logits[row * n_classes..(row + 1) * n_classes].copy_from_slice(&self.init_);
        }

        for stage in 0..self.n_estimators {
            let proba = softmax_rows(&logits, rows, n_classes);
            let sample_rows = sampled_rows(
                rows,
                self.subsample,
                self.random_state.unwrap_or(0) + stage as u64,
            );
            let trees = (0..n_classes)
                .into_par_iter()
                .map(|class| {
                    let residuals = (0..rows)
                        .map(|row| if encoded.encoded[row] == class { 1.0 } else { 0.0 } - proba[row * n_classes + class])
                        .collect::<Vec<_>>();
                    build_regression_tree(x, &residuals, params, stage as u64 * 37 + class as u64, &sample_rows)
                })
                .collect::<Vec<_>>();
            update_logits(&mut logits, x, cols, n_classes, self.learning_rate, &trees);
            self.stages.push(trees);
        }

        self.classes_ = Some(Array::array(&encoded.classes));
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    pub fn decision_function(&self, x: &Array) -> Result<Array, TreeError> {
        let expected = self.n_features_in_.ok_or(TreeError::NotFitted)?;
        validate_predict_input(x, expected)?;
        let n_classes = self.classes_.as_ref().ok_or(TreeError::NotFitted)?.len();
        let rows = x.shape()[0];
        let cols = x.shape()[1];
        let mut logits = vec![0.0; rows * n_classes];
        for row in 0..rows {
            logits[row * n_classes..(row + 1) * n_classes].copy_from_slice(&self.init_);
        }
        for trees in &self.stages {
            update_logits(&mut logits, x, cols, n_classes, self.learning_rate, trees);
        }
        Ok(Array::from_shape_vec(&[rows, n_classes], logits))
    }

    pub fn predict_proba(&self, x: &Array) -> Result<Array, TreeError> {
        let scores = self.decision_function(x)?;
        Ok(Array::from_shape_vec(
            scores.shape(),
            softmax_rows(scores.data(), scores.shape()[0], scores.shape()[1]),
        ))
    }

    pub fn predict_log_proba(&self, x: &Array) -> Result<Array, TreeError> {
        Ok(self.predict_proba(x)?.log())
    }

    pub fn predict(&self, x: &Array) -> Result<Array, TreeError> {
        let proba = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().ok_or(TreeError::NotFitted)?;
        Ok(predicted_classes(&proba, classes))
    }

    pub fn score(&self, x: &Array, y: &Array) -> Result<f64, TreeError> {
        let prediction = self.predict(x)?;
        crate::metrics::accuracy_score(y, &prediction)
            .map_err(|_| TreeError::InvalidLabelShape(y.shape().to_vec()))
    }
}

impl crate::metrics::SupervisedEstimator for GradientBoostingMachine {
    type Error = TreeError;
    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error> {
        GradientBoostingMachine::fit(self, x, y)
    }
    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error> {
        GradientBoostingMachine::score(self, x, y)
    }
}

fn validate_params(
    learning_rate: f64,
    n_estimators: usize,
    subsample: f64,
    max_depth: usize,
    min_split: usize,
    min_leaf: usize,
) -> Result<(), TreeError> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(TreeError::InvalidLearningRate(learning_rate));
    }
    if n_estimators == 0 {
        return Err(TreeError::InvalidEstimatorCount(0));
    }
    if !subsample.is_finite() || subsample <= 0.0 || subsample > 1.0 {
        return Err(TreeError::InvalidSubsample(subsample));
    }
    if max_depth == 0 {
        return Err(TreeError::InvalidMaxDepth(0));
    }
    if min_split < 2 {
        return Err(TreeError::InvalidMinSamplesSplit(min_split));
    }
    if min_leaf == 0 {
        return Err(TreeError::InvalidMinSamplesLeaf(min_leaf));
    }
    Ok(())
}
