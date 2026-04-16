use rayon::prelude::*;

use crate::darray::Array;

use super::common::{
    ClassifierParams, ClassifierTree, Criterion, MaxFeatures, TreeError, average_importances,
    build_classifier_tree, encode_labels, ensemble_proba, predicted_classes, sampled_dataset,
    validate_ensemble_params, validate_fit_inputs, validate_predict_input,
};

#[derive(Debug, Clone, PartialEq)]
pub struct RandomForest {
    pub n_estimators: usize,
    pub criterion: Criterion,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
    pub bootstrap: bool,
    pub random_state: Option<u64>,
    pub classes_: Option<Array>,
    pub feature_importances_: Option<Array>,
    pub n_features_in_: Option<usize>,
    trees: Vec<ClassifierTree>,
}

impl Default for RandomForest {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            criterion: Criterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            bootstrap: true,
            random_state: None,
            classes_: None,
            feature_importances_: None,
            n_features_in_: None,
            trees: Vec::new(),
        }
    }
}

impl RandomForest {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn n_estimators(mut self, value: usize) -> Self {
        self.n_estimators = value;
        self
    }
    pub fn criterion(mut self, value: Criterion) -> Self {
        self.criterion = value;
        self
    }
    pub fn max_depth(mut self, value: usize) -> Self {
        self.max_depth = Some(value);
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
    pub fn bootstrap(mut self, value: bool) -> Self {
        self.bootstrap = value;
        self
    }
    pub fn random_state(mut self, value: u64) -> Self {
        self.random_state = Some(value);
        self
    }
    pub fn is_fitted(&self) -> bool {
        !self.trees.is_empty() && self.classes_.is_some()
    }
    pub fn classes(&self) -> Result<&Array, TreeError> {
        self.classes_.as_ref().ok_or(TreeError::NotFitted)
    }
    pub fn feature_importances(&self) -> Result<&Array, TreeError> {
        self.feature_importances_
            .as_ref()
            .ok_or(TreeError::NotFitted)
    }

    pub fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, TreeError> {
        self.fit_with_params(x, y, self.params(false))
    }

    pub(crate) fn fit_with_params(
        &mut self,
        x: &Array,
        y: &Array,
        params: ClassifierParams,
    ) -> Result<&mut Self, TreeError> {
        validate_fit_inputs(x, y)?;
        validate_ensemble_params(
            self.n_estimators,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        )?;
        let encoded = encode_labels(y)?;
        let base_seed = self.random_state.unwrap_or(0);

        self.trees = (0..self.n_estimators)
            .into_par_iter()
            .map(|index| {
                let seed = base_seed + index as u64;
                let (sampled_x, sampled_y) =
                    sampled_dataset(x, &encoded.encoded, self.bootstrap, seed);
                build_classifier_tree(&sampled_x, &sampled_y, encoded.classes.len(), params, seed)
            })
            .collect();

        self.classes_ = Some(Array::array(&encoded.classes));
        self.feature_importances_ = Some(average_importances(&self.trees, x.shape()[1]));
        self.n_features_in_ = Some(x.shape()[1]);
        Ok(self)
    }

    pub fn predict_proba(&self, x: &Array) -> Result<Array, TreeError> {
        let expected = self.n_features_in_.ok_or(TreeError::NotFitted)?;
        validate_predict_input(x, expected)?;
        let classes = self.classes_.as_ref().ok_or(TreeError::NotFitted)?;
        Ok(ensemble_proba(&self.trees, x, classes.len()))
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

    fn params(&self, random_split: bool) -> ClassifierParams {
        ClassifierParams {
            criterion: self.criterion,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            random_split,
        }
    }
}

impl crate::metrics::SupervisedEstimator for RandomForest {
    type Error = TreeError;
    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error> {
        RandomForest::fit(self, x, y)
    }
    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error> {
        RandomForest::score(self, x, y)
    }
}
