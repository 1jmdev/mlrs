mod error;
mod prediction;
mod training;
mod validation;

pub use error::LinearModelError;
pub(crate) use prediction::predict_from_parameters;
pub(crate) use training::{
    finalize_parameters, fit_elastic_net_coefficients, fit_lasso_coefficients,
    fit_linear_coefficients, fit_logistic_parameters, fit_ridge_coefficients,
    prepare_training_data,
};
pub(crate) use validation::{prepare_class_labels, prepare_targets, validate_features};
