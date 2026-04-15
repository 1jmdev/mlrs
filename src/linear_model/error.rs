use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone, PartialEq)]
pub enum LinearModelError {
    EmptyInput,
    NotFitted,
    InvalidEpochs(usize),
    InvalidLearningRate(f64),
    InvalidFeatureMatrixShape(Vec<usize>),
    InvalidTargetShape(Vec<usize>),
    SampleCountMismatch { x_samples: usize, y_samples: usize },
    FeatureCountMismatch { expected: usize, got: usize },
    SingularMatrix,
}

impl Display for LinearModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::EmptyInput => write!(f, "fit() requires at least one sample and one feature"),
            Self::NotFitted => write!(f, "this LinearRegression instance is not fitted yet"),
            Self::InvalidEpochs(epochs) => {
                write!(f, "epochs must be at least 1, got {epochs}")
            }
            Self::InvalidLearningRate(learning_rate) => write!(
                f,
                "learning_rate must be finite and > 0, got {learning_rate}"
            ),
            Self::InvalidFeatureMatrixShape(shape) => {
                write!(f, "expected X to be 2-D, got shape {shape:?}")
            }
            Self::InvalidTargetShape(shape) => {
                write!(f, "expected y to be 1-D or 2-D, got shape {shape:?}")
            }
            Self::SampleCountMismatch {
                x_samples,
                y_samples,
            } => write!(
                f,
                "X and y have inconsistent numbers of samples: {x_samples} != {y_samples}"
            ),
            Self::FeatureCountMismatch { expected, got } => write!(
                f,
                "X has {got} features, but LinearRegression was fitted with {expected}"
            ),
            Self::SingularMatrix => write!(
                f,
                "least-squares system is singular and could not be solved"
            ),
        }
    }
}

impl Error for LinearModelError {}
