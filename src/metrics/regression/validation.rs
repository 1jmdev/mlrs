use crate::darray::Array;

use super::super::MetricsError;
use super::context::RegressionContext;

/// Validates the base regression input shapes.
pub(crate) fn validate_regression_shapes(
    y_true: &Array,
    y_pred: &Array,
) -> Result<(), MetricsError> {
    if y_true.ndim() == 0 || y_true.ndim() > 2 {
        return Err(MetricsError::InvalidInputShape(y_true.shape().to_vec()));
    }
    if y_pred.ndim() == 0 || y_pred.ndim() > 2 {
        return Err(MetricsError::InvalidInputShape(y_pred.shape().to_vec()));
    }
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::ShapeMismatch {
            y_true: y_true.shape().to_vec(),
            y_pred: y_pred.shape().to_vec(),
        });
    }
    if y_true.shape()[0] == 0 {
        return Err(MetricsError::EmptyInput);
    }

    Ok(())
}

/// Validates that an array contains only finite values.
pub(crate) fn validate_finite_array(array: &Array, name: &'static str) -> Result<(), MetricsError> {
    if array.data().iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(MetricsError::NonFiniteInput(name))
    }
}

/// Validates the non-negative domain required by logarithmic metrics.
pub(crate) fn validate_non_negative_shifted_inputs(
    context: &RegressionContext<'_>,
    metric: &'static str,
) -> Result<(), MetricsError> {
    for sample in 0..context.samples {
        for output in 0..context.outputs {
            if context.y_true_at(sample, output) < 0.0 || context.y_pred_at(sample, output) < 0.0 {
                return Err(MetricsError::InvalidDomain {
                    metric,
                    details: "y_true >= 0 and y_pred >= 0",
                });
            }
        }
    }

    Ok(())
}

/// Validates the Poisson deviance domain.
pub(crate) fn validate_poisson_inputs(context: &RegressionContext<'_>) -> Result<(), MetricsError> {
    for sample in 0..context.samples {
        for output in 0..context.outputs {
            if context.y_true_at(sample, output) < 0.0 || context.y_pred_at(sample, output) <= 0.0 {
                return Err(MetricsError::InvalidDomain {
                    metric: "mean_poisson_deviance",
                    details: "y_true >= 0 and y_pred > 0",
                });
            }
        }
    }

    Ok(())
}

/// Validates the Gamma deviance domain.
pub(crate) fn validate_gamma_inputs(context: &RegressionContext<'_>) -> Result<(), MetricsError> {
    for sample in 0..context.samples {
        for output in 0..context.outputs {
            if context.y_true_at(sample, output) <= 0.0 || context.y_pred_at(sample, output) <= 0.0
            {
                return Err(MetricsError::InvalidDomain {
                    metric: "mean_gamma_deviance",
                    details: "y_true > 0 and y_pred > 0",
                });
            }
        }
    }

    Ok(())
}

/// Validates the Tweedie deviance domain for the requested power.
pub(crate) fn validate_tweedie_inputs(
    context: &RegressionContext<'_>,
    power: f64,
) -> Result<(), MetricsError> {
    if !power.is_finite() {
        return Err(MetricsError::InvalidDomain {
            metric: "mean_tweedie_deviance",
            details: "a finite power value",
        });
    }
    if power > 0.0 && power < 1.0 {
        return Err(MetricsError::InvalidDomain {
            metric: "mean_tweedie_deviance",
            details: "power <= 0 or power >= 1",
        });
    }

    for sample in 0..context.samples {
        for output in 0..context.outputs {
            let true_value = context.y_true_at(sample, output);
            let pred_value = context.y_pred_at(sample, output);

            if power <= 0.0 {
                if pred_value <= 0.0 {
                    return Err(MetricsError::InvalidDomain {
                        metric: "mean_tweedie_deviance",
                        details: "y_pred > 0 when power <= 0",
                    });
                }
            } else if power < 2.0 {
                if true_value < 0.0 || pred_value <= 0.0 {
                    return Err(MetricsError::InvalidDomain {
                        metric: "mean_tweedie_deviance",
                        details: "y_true >= 0 and y_pred > 0 when 1 <= power < 2",
                    });
                }
            } else if true_value <= 0.0 || pred_value <= 0.0 {
                return Err(MetricsError::InvalidDomain {
                    metric: "mean_tweedie_deviance",
                    details: "y_true > 0 and y_pred > 0 when power >= 2",
                });
            }
        }
    }

    Ok(())
}
