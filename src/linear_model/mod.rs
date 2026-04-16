mod classification;
mod common;
mod regression;

pub use classification::LogisticRegression;
pub use common::LinearModelError;
pub use regression::{ElasticNet, Lasso, LinearRegression, Ridge};
