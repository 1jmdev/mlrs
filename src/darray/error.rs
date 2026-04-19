use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone, PartialEq)]
pub enum DArrayError {
    DotShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    DotUnsupportedDimensions { left: Vec<usize>, right: Vec<usize> },
    MatmulInvalidLeftShape(Vec<usize>),
    MatmulInvalidRightShape(Vec<usize>),
    MatmulShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    OuterInvalidLeftShape(Vec<usize>),
    OuterInvalidRightShape(Vec<usize>),
    VdotShapeMismatch { left_len: usize, right_len: usize },
    DiagInvalidShape(Vec<usize>),
    TraceInvalidShape(Vec<usize>),
    InvalidUniformRange { low: f64, high: f64 },
    InvalidRandintRange { low: i64, high: i64 },
    EmptyInput(&'static str),
    ChoiceSampleTooLarge { requested: usize, available: usize },
}

impl Display for DArrayError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::DotShapeMismatch { left, right } => {
                write!(f, "dot() shape mismatch: {left:?} and {right:?}")
            }
            Self::DotUnsupportedDimensions { left, right } => write!(
                f,
                "dot() supports 1-D x 1-D, 2-D x 1-D, and 2-D x 2-D, got {left:?} and {right:?}"
            ),
            Self::MatmulInvalidLeftShape(shape) => {
                write!(f, "matmul() requires a 2-D left operand, got {shape:?}")
            }
            Self::MatmulInvalidRightShape(shape) => {
                write!(f, "matmul() requires a 2-D right operand, got {shape:?}")
            }
            Self::MatmulShapeMismatch { left, right } => {
                write!(f, "matmul() shape mismatch: {left:?} and {right:?}")
            }
            Self::OuterInvalidLeftShape(shape) => {
                write!(f, "outer() requires a 1-D left operand, got {shape:?}")
            }
            Self::OuterInvalidRightShape(shape) => {
                write!(f, "outer() requires a 1-D right operand, got {shape:?}")
            }
            Self::VdotShapeMismatch { left_len, right_len } => {
                write!(f, "vdot() shape mismatch: {left_len} and {right_len}")
            }
            Self::DiagInvalidShape(shape) => {
                write!(f, "diag() requires a 1-D or 2-D array, got {shape:?}")
            }
            Self::TraceInvalidShape(shape) => {
                write!(f, "trace() requires a 2-D array, got {shape:?}")
            }
            Self::InvalidUniformRange { low, high } => {
                write!(f, "uniform() requires low < high, got low={low}, high={high}")
            }
            Self::InvalidRandintRange { low, high } => {
                write!(f, "randint() requires low < high, got low={low}, high={high}")
            }
            Self::EmptyInput(name) => write!(f, "{name} requires at least one value"),
            Self::ChoiceSampleTooLarge {
                requested,
                available,
            } => write!(
                f,
                "choice(replace=false) requires size <= number of values, got size={requested}, values={available}"
            ),
        }
    }
}

impl Error for DArrayError {}
