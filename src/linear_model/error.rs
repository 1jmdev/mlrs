use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Describes the validation and runtime errors raised by linear models.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearModelError {
    /// The provided inputs contain no samples or no features.
    EmptyInput,
    /// The model was used before fitting coefficients and intercepts.
    NotFitted,
    /// The configured epoch count is invalid.
    InvalidEpochs(usize),
    /// The configured learning rate is not finite or positive.
    InvalidLearningRate(f64),
    /// The feature matrix does not have two dimensions.
    InvalidFeatureMatrixShape(Vec<usize>),
    /// The target array is neither a vector nor a matrix.
    InvalidTargetShape(Vec<usize>),
    /// The feature matrix and target array disagree on sample count.
    SampleCountMismatch { x_samples: usize, y_samples: usize },
    /// Prediction input used a different feature count than training.
    FeatureCountMismatch { expected: usize, got: usize },
    /// The least-squares system could not be solved safely.
    SingularMatrix,
}

impl Display for LinearModelError {
    /// Formats the error in a user-facing sklearn-style message.
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

/// Marks `LinearModelError` as a standard error type.
impl Error for LinearModelError {}
