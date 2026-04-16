use crate::darray::Array;

use super::common::{
    ClassifierParams, ClassifierTree, Criterion, MaxFeatures, TreeError, build_classifier_tree,
    encode_labels, predicted_classes, tree_proba_matrix, validate_fit_inputs,
    validate_predict_input,
};

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTree {
    pub criterion: Criterion,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
    pub classes_: Option<Array>,
    pub feature_importances_: Option<Array>,
    pub n_features_in_: Option<usize>,
    tree: Option<ClassifierTree>,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self {
            criterion: Criterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            classes_: None,
            feature_importances_: None,
            n_features_in_: None,
            tree: None,
        }
    }
}

impl DecisionTree {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
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
    pub fn is_fitted(&self) -> bool {
        self.tree.is_some() && self.classes_.is_some()
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
        validate_fit_inputs(x, y)?;
        validate_params(
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_features,
            x.shape()[1],
        )?;
        let encoded = encode_labels(y)?;
        let tree =
            build_classifier_tree(x, &encoded.encoded, encoded.classes.len(), self.params(), 0);

        self.classes_ = Some(Array::array(&encoded.classes));
        self.feature_importances_ = Some(normalize_importances(tree.feature_importances.clone()));
        self.n_features_in_ = Some(x.shape()[1]);
        self.tree = Some(tree);
        Ok(self)
    }

    pub fn predict_proba(&self, x: &Array) -> Result<Array, TreeError> {
        let expected = self.n_features_in_.ok_or(TreeError::NotFitted)?;
        validate_predict_input(x, expected)?;
        let tree = self.tree.as_ref().ok_or(TreeError::NotFitted)?;
        let n_classes = self.classes_.as_ref().ok_or(TreeError::NotFitted)?.len();
        Ok(tree_proba_matrix(tree, x, n_classes))
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

    fn params(&self) -> ClassifierParams {
        ClassifierParams {
            criterion: self.criterion,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            random_split: false,
        }
    }
}

impl crate::metrics::SupervisedEstimator for DecisionTree {
    type Error = TreeError;
    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error> {
        DecisionTree::fit(self, x, y)
    }
    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error> {
        DecisionTree::score(self, x, y)
    }
}

fn validate_params(
    max_depth: Option<usize>,
    min_split: usize,
    min_leaf: usize,
    max_features: MaxFeatures,
    n_features: usize,
) -> Result<(), TreeError> {
    if max_depth == Some(0) {
        return Err(TreeError::InvalidMaxDepth(0));
    }
    if min_split < 2 {
        return Err(TreeError::InvalidMinSamplesSplit(min_split));
    }
    if min_leaf == 0 {
        return Err(TreeError::InvalidMinSamplesLeaf(min_leaf));
    }
    if matches!(max_features, MaxFeatures::Count(0))
        || matches!(max_features, MaxFeatures::Fraction(value) if !value.is_finite() || value <= 0.0)
    {
        return Err(TreeError::InvalidMaxFeatures);
    }
    if n_features == 0 {
        return Err(TreeError::EmptyInput);
    }
    Ok(())
}

fn normalize_importances(mut values: Vec<f64>) -> Array {
    let total = values.iter().sum::<f64>();
    if total > 0.0 {
        values.iter_mut().for_each(|value| *value /= total);
    }
    Array::array(&values)
}
