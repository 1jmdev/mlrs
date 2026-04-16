use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone, PartialEq)]
pub enum TreeError {
    EmptyInput,
    NotFitted,
    InvalidFeatureMatrixShape(Vec<usize>),
    InvalidLabelShape(Vec<usize>),
    SampleCountMismatch { x_samples: usize, y_samples: usize },
    InvalidClassCount(usize),
    InvalidMaxDepth(usize),
    InvalidMinSamplesSplit(usize),
    InvalidMinSamplesLeaf(usize),
    InvalidEstimatorCount(usize),
    InvalidLearningRate(f64),
    InvalidSubsample(f64),
    InvalidMaxFeatures,
    FeatureCountMismatch { expected: usize, got: usize },
}

impl Display for TreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::EmptyInput => write!(f, "fit() requires at least one sample and one feature"),
            Self::NotFitted => write!(f, "this tree-based estimator instance is not fitted yet"),
            Self::InvalidFeatureMatrixShape(shape) => {
                write!(f, "expected X to be 2-D, got shape {shape:?}")
            }
            Self::InvalidLabelShape(shape) => {
                write!(
                    f,
                    "expected y to be a non-empty 1-D array, got shape {shape:?}"
                )
            }
            Self::SampleCountMismatch {
                x_samples,
                y_samples,
            } => write!(
                f,
                "X and y have inconsistent numbers of samples: {x_samples} != {y_samples}"
            ),
            Self::InvalidClassCount(classes) => {
                write!(
                    f,
                    "classification requires at least 2 distinct classes, got {classes}"
                )
            }
            Self::InvalidMaxDepth(depth) => write!(f, "max_depth must be at least 1, got {depth}"),
            Self::InvalidMinSamplesSplit(value) => {
                write!(f, "min_samples_split must be at least 2, got {value}")
            }
            Self::InvalidMinSamplesLeaf(value) => {
                write!(f, "min_samples_leaf must be at least 1, got {value}")
            }
            Self::InvalidEstimatorCount(value) => {
                write!(f, "n_estimators must be at least 1, got {value}")
            }
            Self::InvalidLearningRate(value) => {
                write!(f, "learning_rate must be finite and > 0, got {value}")
            }
            Self::InvalidSubsample(value) => {
                write!(f, "subsample must be finite and in (0, 1], got {value}")
            }
            Self::InvalidMaxFeatures => write!(f, "max_features resolved to zero features"),
            Self::FeatureCountMismatch { expected, got } => {
                write!(
                    f,
                    "X has {got} features, but the model was fitted with {expected}"
                )
            }
        }
    }
}

impl Error for TreeError {}
