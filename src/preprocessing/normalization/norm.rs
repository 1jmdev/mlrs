use super::super::PreprocessingError;

/// Selects the row-wise norm used by `Normalizer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    /// Uses the sum of absolute values.
    L1,
    /// Uses the Euclidean norm.
    L2,
    /// Uses the maximum absolute value.
    Max,
}

impl Norm {
    /// Parses a sklearn-style norm string.
    pub fn from_str(name: &str) -> Result<Self, PreprocessingError> {
        match name {
            "l1" => Ok(Self::L1),
            "l2" => Ok(Self::L2),
            "max" => Ok(Self::Max),
            _ => Err(PreprocessingError::InvalidNorm("unknown")),
        }
    }
}
