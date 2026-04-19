mod error;
mod math;
mod prediction;
mod training;
mod validation;

pub use error::LinearModelError;
pub(crate) use math::{
    cholesky_decompose, cholesky_solve, compute_gram_and_xty, dot_simd, scaled_add_assign,
    scaled_sub_assign, soft_threshold, subtract_in_place,
};
pub(crate) use prediction::predict_from_parameters;
pub(crate) use training::{finalize_parameters, prepare_training_data};
pub(crate) use validation::{prepare_targets, validate_features};
