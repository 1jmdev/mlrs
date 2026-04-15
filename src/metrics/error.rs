use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Errors returned by metric computations.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricsError {
    EmptyInput,
    InvalidInputShape(Vec<usize>),
    ShapeMismatch {
        y_true: Vec<usize>,
        y_pred: Vec<usize>,
    },
    InvalidSampleWeightShape(Vec<usize>),
    SampleWeightLengthMismatch {
        expected: usize,
        got: usize,
    },
    NonFiniteInput(&'static str),
    InvalidMultiOutputWeights {
        expected: usize,
        got: usize,
    },
    UnsupportedMultiOutput(&'static str),
    InvalidAlpha(f64),
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
            Self::ShapeMismatch { y_true, y_pred } => write!(
                f,
                "y_true and y_pred must have the same shape, got {y_true:?} and {y_pred:?}"
            ),
            Self::InvalidSampleWeightShape(shape) => {
                write!(f, "sample_weight must be 1-D, got shape {shape:?}")
            }
            Self::SampleWeightLengthMismatch { expected, got } => write!(
                f,
                "sample_weight must contain one weight per sample: expected {expected}, got {got}"
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
            Self::InvalidDomain { metric, details } => write!(f, "{metric} requires {details}"),
        }
    }
}

impl Error for MetricsError {}
