use crate::darray::Array;

use super::super::MetricsError;
use super::types::SplitSize;

pub(crate) fn validate_split_inputs(x: &Array, y: &Array) -> Result<usize, MetricsError> {
    if !x.is_matrix() || x.shape()[0] == 0 || x.shape()[1] == 0 {
        return Err(MetricsError::InvalidInputShape(x.shape().to_vec()));
    }
    if y.ndim() == 0 || y.ndim() > 2 || y.shape()[0] == 0 {
        return Err(MetricsError::InvalidInputShape(y.shape().to_vec()));
    }
    let x_samples = x.shape()[0];
    let y_samples = y.shape()[0];
    if x_samples != y_samples {
        return Err(MetricsError::SampleCountMismatch {
            x_samples,
            y_samples,
        });
    }
    if x.data().iter().any(|value| !value.is_finite()) {
        return Err(MetricsError::NonFiniteInput("X"));
    }
    if y.data().iter().any(|value| !value.is_finite()) {
        return Err(MetricsError::NonFiniteInput("y"));
    }
    Ok(x_samples)
}

pub(crate) fn resolve_split_sizes(
    samples: usize,
    train_size: Option<SplitSize>,
    test_size: Option<SplitSize>,
) -> Result<(usize, usize), MetricsError> {
    let test = match test_size {
        Some(size) => resolve_size(samples, size, "test_size")?,
        None => 0,
    };
    let train = match train_size {
        Some(size) => resolve_size(samples, size, "train_size")?,
        None => samples.saturating_sub(test),
    };

    if train == 0 {
        return Err(MetricsError::InvalidSplitSize {
            name: "train_size",
            details: "resolved to zero samples",
        });
    }
    if test == 0 {
        return Err(MetricsError::InvalidSplitSize {
            name: "test_size",
            details: "resolved to zero samples",
        });
    }
    if train + test > samples {
        return Err(MetricsError::InvalidSplitSize {
            name: "train_size/test_size",
            details: "resolved sizes exceed the number of samples",
        });
    }
    if train_size.is_some() && test_size.is_some() && train + test != samples {
        return Err(MetricsError::InvalidSplitSize {
            name: "train_size/test_size",
            details: "when both are set, resolved sizes must sum to n_samples",
        });
    }
    Ok((train, test))
}

pub(crate) fn resolve_size(
    samples: usize,
    size: SplitSize,
    name: &'static str,
) -> Result<usize, MetricsError> {
    match size {
        SplitSize::Count(count) => {
            if count == 0 || count >= samples {
                Err(MetricsError::InvalidSplitSize {
                    name,
                    details: "count must be in [1, n_samples - 1]",
                })
            } else {
                Ok(count)
            }
        }
        SplitSize::Ratio(ratio) => {
            if !ratio.is_finite() || ratio <= 0.0 || ratio >= 1.0 {
                return Err(MetricsError::InvalidSplitSize {
                    name,
                    details: "ratio must be finite and in (0, 1)",
                });
            }
            let count = (samples as f64 * ratio).round() as usize;
            if count == 0 || count >= samples {
                Err(MetricsError::InvalidSplitSize {
                    name,
                    details: "ratio resolved to an empty split",
                })
            } else {
                Ok(count)
            }
        }
    }
}
