use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Errors returned by preprocessing transformers.
#[derive(Debug, Clone, PartialEq)]
pub enum PreprocessingError {
    /// The input array must be a non-empty 2-D matrix.
    InvalidInputShape(Vec<usize>),
    /// The label array must be a non-empty 1-D vector.
    InvalidLabelShape(Vec<usize>),
    /// The transformer was used before `fit` populated learned parameters.
    NotFitted(&'static str),
    /// The input feature count does not match the fitted feature count.
    FeatureCountMismatch { expected: usize, got: usize },
    /// Inputs must contain finite floating-point values.
    NonFiniteInput(&'static str),
    /// The requested feature range is invalid.
    InvalidFeatureRange { min: f64, max: f64 },
    /// The requested quantile range is invalid.
    InvalidQuantileRange { lower: f64, upper: f64 },
    /// The requested norm is not defined.
    InvalidNorm(&'static str),
    /// A manual category specification does not match the input shape.
    InvalidCategories { expected: usize, got: usize },
    /// Transform encountered a category that was not seen during fitting.
    UnknownCategory { feature_index: usize, value: f64 },
    /// Transform encountered a label that was not seen during fitting.
    UnknownLabel(f64),
    /// Ordinal encoding requires an explicit unknown value sentinel.
    MissingUnknownValue,
    /// The configured unknown value collides with a fitted category index.
    InvalidUnknownValue(f64),
    /// Inverse transform received an invalid encoded label.
    InvalidEncodedLabel(f64),
    /// Inverse transform could not decode a one-hot segment safely.
    InvalidEncodedRow {
        sample_index: usize,
        feature_index: usize,
        details: &'static str,
    },
    /// A configured constant fill value is not finite.
    InvalidFillValue(f64),
    /// A requested imputation statistic could not be computed for a feature.
    MissingStatistic {
        feature_index: usize,
        strategy: &'static str,
    },
}

impl Display for PreprocessingError {
    /// Formats the error with sklearn-style user-facing wording.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidInputShape(shape) => {
                write!(f, "expected a non-empty 2-D array, got shape {shape:?}")
            }
            Self::InvalidLabelShape(shape) => {
                write!(
                    f,
                    "expected a non-empty 1-D label array, got shape {shape:?}"
                )
            }
            Self::NotFitted(name) => write!(f, "this {name} instance is not fitted yet"),
            Self::FeatureCountMismatch { expected, got } => write!(
                f,
                "X has {got} features, but the transformer was fitted with {expected}"
            ),
            Self::NonFiniteInput(name) => write!(f, "{name} must contain only finite values"),
            Self::InvalidFeatureRange { min, max } => {
                write!(f, "feature_range min must be < max, got ({min}, {max})")
            }
            Self::InvalidQuantileRange { lower, upper } => write!(
                f,
                "quantile_range must satisfy 0 < lower < upper < 100, got ({lower}, {upper})"
            ),
            Self::InvalidNorm(norm) => {
                write!(f, "unsupported norm {norm}; expected 'l1', 'l2', or 'max'")
            }
            Self::InvalidCategories { expected, got } => write!(
                f,
                "manual categories must provide one category list per feature: expected {expected}, got {got}"
            ),
            Self::UnknownCategory {
                feature_index,
                value,
            } => write!(
                f,
                "found unknown category {value} in feature {feature_index} during transform"
            ),
            Self::UnknownLabel(value) => {
                write!(f, "y contains previously unseen label {value}")
            }
            Self::MissingUnknownValue => write!(
                f,
                "handle_unknown='use_encoded_value' requires unknown_value to be set"
            ),
            Self::InvalidUnknownValue(value) => write!(
                f,
                "unknown_value must not collide with fitted category indices, got {value}"
            ),
            Self::InvalidEncodedLabel(value) => {
                write!(f, "encoded label {value} is not a valid fitted class index")
            }
            Self::InvalidEncodedRow {
                sample_index,
                feature_index,
                details,
            } => write!(
                f,
                "could not decode sample {sample_index}, feature {feature_index}: {details}"
            ),
            Self::InvalidFillValue(fill_value) => {
                write!(f, "fill_value must be finite, got {fill_value}")
            }
            Self::MissingStatistic {
                feature_index,
                strategy,
            } => write!(
                f,
                "cannot compute {strategy} for feature {feature_index} because it contains only missing values"
            ),
        }
    }
}

impl Error for PreprocessingError {}
