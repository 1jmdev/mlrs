#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Criterion {
    Gini,
    Entropy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
    Count(usize),
    Fraction(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ClassifierParams {
    pub(crate) criterion: Criterion,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) max_features: MaxFeatures,
    pub(crate) random_split: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GradientBoostingParams {
    pub(crate) max_depth: usize,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) max_features: MaxFeatures,
}

pub(crate) fn resolved_max_features(max_features: MaxFeatures, n_features: usize) -> usize {
    match max_features {
        MaxFeatures::All => n_features,
        MaxFeatures::Sqrt => (n_features as f64).sqrt().floor().max(1.0) as usize,
        MaxFeatures::Log2 => (n_features as f64).log2().floor().max(1.0) as usize,
        MaxFeatures::Count(value) => value.min(n_features),
        MaxFeatures::Fraction(value) => ((n_features as f64) * value).floor().max(1.0) as usize,
    }
}
