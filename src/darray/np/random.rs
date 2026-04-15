use crate::darray::{Array, RandomState};

pub use crate::darray::RandomState as Generator;

/// Creates a fresh RNG seeded from the operating system.
pub fn default_rng() -> RandomState {
    RandomState::new()
}

/// Creates a seeded RNG.
pub fn seeded(seed: u64) -> RandomState {
    RandomState::seeded(seed)
}

/// Returns uniform random samples in `[0, 1)`.
pub fn random(shape: &[usize]) -> Array {
    let mut rng = RandomState::new();
    rng.random(shape)
}

/// Returns uniform random samples in `[0, 1)`.
pub fn random_sample(shape: &[usize]) -> Array {
    let mut rng = RandomState::new();
    rng.random_sample(shape)
}

/// Returns uniform random samples in `[0, 1)`.
pub fn rand(shape: &[usize]) -> Array {
    let mut rng = RandomState::new();
    rng.rand(shape)
}

/// Returns standard normal random samples.
pub fn randn(shape: &[usize]) -> Array {
    let mut rng = RandomState::new();
    rng.randn(shape)
}

/// Returns normal random samples.
pub fn normal(mean: f64, std: f64, shape: &[usize]) -> Array {
    let mut rng = RandomState::new();
    rng.normal(mean, std, shape)
}

/// Returns uniform random samples in `[low, high)`.
pub fn uniform(low: f64, high: f64, shape: &[usize]) -> Array {
    let mut rng = RandomState::new();
    rng.uniform(low, high, shape)
}

/// Returns random integers in `[low, high)` as `f64` values.
pub fn randint(low: i64, high: i64, shape: &[usize]) -> Array {
    let mut rng = RandomState::new();
    rng.randint(low, high, shape)
}

/// Chooses values from a flattened input array.
pub fn choice(values: &Array, size: usize, replace: bool) -> Array {
    let mut rng = RandomState::new();
    rng.choice(values, size, replace)
}

/// Returns a shuffled copy of the input array.
pub fn permutation(values: &Array) -> Array {
    let mut rng = RandomState::new();
    rng.permutation(values)
}
