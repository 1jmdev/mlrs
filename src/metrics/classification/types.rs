use crate::darray::Array;

/// Controls how per-class classification metrics are aggregated.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationAverage {
    Binary,
    Micro,
    Macro,
    Weighted,
    None,
}

/// Optional arguments for accuracy scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyOptions<'a> {
    pub normalize: bool,
    pub sample_weight: Option<&'a Array>,
}

impl<'a> Default for AccuracyOptions<'a> {
    fn default() -> Self {
        Self {
            normalize: true,
            sample_weight: None,
        }
    }
}

impl<'a> AccuracyOptions<'a> {
    /// Creates default accuracy options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to return a fraction or a weighted count.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Sets optional per-sample weights.
    pub fn with_sample_weight(mut self, sample_weight: &'a Array) -> Self {
        self.sample_weight = Some(sample_weight);
        self
    }
}

/// Optional arguments for precision, recall, and F1.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationMetricOptions<'a> {
    pub labels: Option<&'a Array>,
    pub sample_weight: Option<&'a Array>,
    pub average: ClassificationAverage,
    pub pos_label: f64,
    pub zero_division: f64,
}

impl<'a> Default for ClassificationMetricOptions<'a> {
    fn default() -> Self {
        Self {
            labels: None,
            sample_weight: None,
            average: ClassificationAverage::Binary,
            pos_label: 1.0,
            zero_division: 0.0,
        }
    }
}

impl<'a> ClassificationMetricOptions<'a> {
    /// Creates default classification-metric options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets explicit class labels.
    pub fn with_labels(mut self, labels: &'a Array) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Sets optional per-sample weights.
    pub fn with_sample_weight(mut self, sample_weight: &'a Array) -> Self {
        self.sample_weight = Some(sample_weight);
        self
    }

    /// Sets the class-aggregation policy.
    pub fn with_average(mut self, average: ClassificationAverage) -> Self {
        self.average = average;
        self
    }

    /// Sets the positive label used for binary metrics.
    pub fn with_pos_label(mut self, pos_label: f64) -> Self {
        self.pos_label = pos_label;
        self
    }

    /// Sets the value returned for zero-division cases.
    pub fn with_zero_division(mut self, zero_division: f64) -> Self {
        self.zero_division = zero_division;
        self
    }
}

/// Returns either a scalar metric or one value per class.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationMetricOutput {
    Scalar(f64),
    PerClass(Array),
}

impl ClassificationMetricOutput {
    /// Returns the scalar value when the metric was aggregated.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Self::Scalar(value) => Some(*value),
            Self::PerClass(_) => None,
        }
    }

    /// Returns the per-class values when requested.
    pub fn as_per_class(&self) -> Option<&Array> {
        match self {
            Self::Scalar(_) => None,
            Self::PerClass(values) => Some(values),
        }
    }
}

/// Controls how confusion matrices are normalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfusionMatrixNormalize {
    None,
    True,
    Pred,
    All,
}

/// Optional arguments for confusion-matrix construction.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfusionMatrixOptions<'a> {
    pub labels: Option<&'a Array>,
    pub sample_weight: Option<&'a Array>,
    pub normalize: ConfusionMatrixNormalize,
}

impl<'a> Default for ConfusionMatrixOptions<'a> {
    fn default() -> Self {
        Self {
            labels: None,
            sample_weight: None,
            normalize: ConfusionMatrixNormalize::None,
        }
    }
}

impl<'a> ConfusionMatrixOptions<'a> {
    /// Creates default confusion-matrix options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets explicit class labels.
    pub fn with_labels(mut self, labels: &'a Array) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Sets optional per-sample weights.
    pub fn with_sample_weight(mut self, sample_weight: &'a Array) -> Self {
        self.sample_weight = Some(sample_weight);
        self
    }

    /// Sets the normalization mode.
    pub fn with_normalize(mut self, normalize: ConfusionMatrixNormalize) -> Self {
        self.normalize = normalize;
        self
    }
}
