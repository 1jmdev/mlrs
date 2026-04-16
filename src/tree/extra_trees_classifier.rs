use crate::darray::Array;

use super::common::{ClassifierParams, Criterion, MaxFeatures, TreeError};
use super::random_forest::RandomForest;

#[derive(Debug, Clone, PartialEq)]
pub struct ExtraTreesClassifier {
    forest: RandomForest,
}

impl Default for ExtraTreesClassifier {
    fn default() -> Self {
        Self {
            forest: RandomForest::new().bootstrap(false),
        }
    }
}

impl ExtraTreesClassifier {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn n_estimators(mut self, value: usize) -> Self {
        self.forest.n_estimators = value;
        self
    }
    pub fn criterion(mut self, value: Criterion) -> Self {
        self.forest.criterion = value;
        self
    }
    pub fn max_depth(mut self, value: usize) -> Self {
        self.forest.max_depth = Some(value);
        self
    }
    pub fn min_samples_split(mut self, value: usize) -> Self {
        self.forest.min_samples_split = value;
        self
    }
    pub fn min_samples_leaf(mut self, value: usize) -> Self {
        self.forest.min_samples_leaf = value;
        self
    }
    pub fn max_features(mut self, value: MaxFeatures) -> Self {
        self.forest.max_features = value;
        self
    }
    pub fn random_state(mut self, value: u64) -> Self {
        self.forest.random_state = Some(value);
        self
    }
    pub fn is_fitted(&self) -> bool {
        self.forest.is_fitted()
    }
    pub fn classes(&self) -> Result<&Array, TreeError> {
        self.forest.classes_.as_ref().ok_or(TreeError::NotFitted)
    }
    pub fn feature_importances(&self) -> Result<&Array, TreeError> {
        self.forest
            .feature_importances_
            .as_ref()
            .ok_or(TreeError::NotFitted)
    }

    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, TreeError> {
        self.forest.fit_with_params(x, y, self.params())?;
        Ok(self)
    }

    pub fn predict_proba(&self, x: &Array) -> Result<Array, TreeError> {
        self.forest.predict_proba(x)
    }
    pub fn predict_log_proba(&self, x: &Array) -> Result<Array, TreeError> {
        Ok(self.predict_proba(x)?.log())
    }
    pub fn predict(&self, x: &Array) -> Result<Array, TreeError> {
        self.forest.predict(x)
    }
    pub fn score(&self, x: &Array, y: &Array) -> Result<f64, TreeError> {
        self.forest.score(x, y)
    }

    fn params(&self) -> ClassifierParams {
        ClassifierParams {
            criterion: self.forest.criterion,
            max_depth: self.forest.max_depth,
            min_samples_split: self.forest.min_samples_split,
            min_samples_leaf: self.forest.min_samples_leaf,
            max_features: self.forest.max_features,
            random_split: true,
        }
    }
}

impl crate::metrics::SupervisedEstimator for ExtraTreesClassifier {
    type Error = TreeError;
    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error> {
        ExtraTreesClassifier::fit(self, x, y)
    }
    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error> {
        ExtraTreesClassifier::score(self, x, y)
    }
}
