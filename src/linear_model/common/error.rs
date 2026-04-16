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
    /// The configured regularization strength is not finite or is negative.
    InvalidAlpha(f64),
    /// The configured coordinate descent tolerance is not finite or positive.
    InvalidTolerance(f64),
    /// The configured iteration count is invalid.
    InvalidMaxIterations(usize),
    /// The configured elastic-net `l1_ratio` is invalid.
    InvalidL1Ratio(f64),
    /// The feature matrix does not have two dimensions.
    InvalidFeatureMatrixShape(Vec<usize>),
    /// The target array is neither a vector nor a matrix.
    InvalidTargetShape(Vec<usize>),
    /// Logistic regression requires a non-empty 1-D label array.
    InvalidLabelShape(Vec<usize>),
    /// The feature matrix and target array disagree on sample count.
    SampleCountMismatch { x_samples: usize, y_samples: usize },
    /// Logistic regression requires at least two distinct classes.
    InvalidClassCount(usize),
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
            Self::NotFitted => write!(f, "this linear model instance is not fitted yet"),
            Self::InvalidEpochs(epochs) => {
                write!(f, "epochs must be at least 1, got {epochs}")
            }
            Self::InvalidLearningRate(learning_rate) => write!(
                f,
                "learning_rate must be finite and > 0, got {learning_rate}"
            ),
            Self::InvalidAlpha(alpha) => {
                write!(f, "alpha must be finite and >= 0, got {alpha}")
            }
            Self::InvalidTolerance(tol) => {
                write!(f, "tol must be finite and > 0, got {tol}")
            }
            Self::InvalidMaxIterations(max_iter) => {
                write!(f, "max_iter must be at least 1, got {max_iter}")
            }
            Self::InvalidL1Ratio(l1_ratio) => {
                write!(f, "l1_ratio must be finite and in [0, 1], got {l1_ratio}")
            }
            Self::InvalidFeatureMatrixShape(shape) => {
                write!(f, "expected X to be 2-D, got shape {shape:?}")
            }
            Self::InvalidTargetShape(shape) => {
                write!(f, "expected y to be 1-D or 2-D, got shape {shape:?}")
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
            Self::InvalidClassCount(classes) => write!(
                f,
                "logistic regression requires at least 2 distinct classes, got {classes}"
            ),
            Self::FeatureCountMismatch { expected, got } => write!(
                f,
                "X has {got} features, but the model was fitted with {expected}"
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
