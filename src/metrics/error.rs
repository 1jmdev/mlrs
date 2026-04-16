use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Errors returned by metric computations.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricsError {
    EmptyInput,
    InvalidInputShape(Vec<usize>),
    InvalidClassificationShape(Vec<usize>),
    ShapeMismatch {
        y_true: Vec<usize>,
        y_pred: Vec<usize>,
    },
    EmptyLabels,
    UnknownLabel(f64),
    InvalidSampleWeightShape(Vec<usize>),
    SampleWeightLengthMismatch {
        expected: usize,
        got: usize,
    },
    InvalidZeroDivision(f64),
    InvalidAverage(&'static str),
    InvalidSplitSize {
        name: &'static str,
        details: &'static str,
    },
    InvalidCv(usize),
    SampleCountMismatch {
        x_samples: usize,
        y_samples: usize,
    },
    NonFiniteInput(&'static str),
    InvalidMultiOutputWeights {
        expected: usize,
        got: usize,
    },
    UnsupportedMultiOutput(&'static str),
    InvalidAlpha(f64),
    EstimatorError(String),
    InvalidDomain {
        metric: &'static str,
        details: &'static str,
    },
}

impl Display for MetricsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::EmptyInput => write!(f, "metric inputs must contain at least one sample"),
            Self::InvalidInputShape(shape) => {
                write!(f, "expected a 1-D or 2-D array, got shape {shape:?}")
            }
            Self::InvalidClassificationShape(shape) => {
                write!(
                    f,
                    "expected a non-empty 1-D label array, got shape {shape:?}"
                )
            }
            Self::ShapeMismatch { y_true, y_pred } => write!(
                f,
                "y_true and y_pred must have the same shape, got {y_true:?} and {y_pred:?}"
            ),
            Self::EmptyLabels => write!(f, "labels must contain at least one class label"),
            Self::UnknownLabel(value) => write!(f, "requested label {value} was not found"),
            Self::InvalidSampleWeightShape(shape) => {
                write!(f, "sample_weight must be 1-D, got shape {shape:?}")
            }
            Self::SampleWeightLengthMismatch { expected, got } => write!(
                f,
                "sample_weight must contain one weight per sample: expected {expected}, got {got}"
            ),
            Self::InvalidZeroDivision(value) => {
                write!(f, "zero_division must be finite, got {value}")
            }
            Self::InvalidAverage(average) => write!(
                f,
                "average='{average}' is incompatible with the provided labels"
            ),
            Self::InvalidSplitSize { name, details } => {
                write!(f, "invalid {name}: {details}")
            }
            Self::InvalidCv(cv) => write!(f, "cv must be at least 2, got {cv}"),
            Self::SampleCountMismatch {
                x_samples,
                y_samples,
            } => write!(
                f,
                "X and y have inconsistent numbers of samples: {x_samples} != {y_samples}"
            ),
            Self::NonFiniteInput(name) => write!(f, "{name} must contain only finite values"),
            Self::InvalidMultiOutputWeights { expected, got } => write!(
                f,
                "multioutput weights must match the number of outputs: expected {expected}, got {got}"
            ),
            Self::UnsupportedMultiOutput(metric) => write!(
                f,
                "{metric} only supports scalar output aggregation and does not support raw values"
            ),
            Self::InvalidAlpha(alpha) => {
                write!(f, "alpha must be finite and in [0, 1], got {alpha}")
            }
            Self::EstimatorError(message) => {
                write!(f, "estimator error during cross validation: {message}")
            }
            Self::InvalidDomain { metric, details } => write!(f, "{metric} requires {details}"),
        }
    }
}

impl Error for MetricsError {}
