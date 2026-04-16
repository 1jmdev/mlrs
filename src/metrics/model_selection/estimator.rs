use crate::darray::Array;

/// Defines the fit/score interface required by `cross_val_score`.
pub trait SupervisedEstimator: Clone {
    type Error;

    fn fit(&mut self, x: &Array, y: &Array) -> Result<&mut Self, Self::Error>;
    fn score(&self, x: &Array, y: &Array) -> Result<f64, Self::Error>;
}
