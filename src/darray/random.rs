use fastrand::Rng;

use super::Array;
use super::utils::compute_size;

pub struct RandomState {
    rng: Rng,
}

impl Default for RandomState {
    fn default() -> Self {
        Self { rng: Rng::new() }
    }
}

impl RandomState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: Rng::with_seed(seed),
        }
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.rng.f64().max(f64::MIN_POSITIVE);
        let u2 = self.rng.f64();
        (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
    }

    fn shuffle<T>(&mut self, data: &mut [T]) {
        for i in (1..data.len()).rev() {
            let j = self.rng.usize(0..=i);
            data.swap(i, j);
        }
    }

    pub fn shuffle_indices(&mut self, indices: &mut [usize]) {
        self.shuffle(indices);
    }

    pub fn random(&mut self, shape: &[usize]) -> Array {
        let size = compute_size(shape);
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.rng.f64());
        }
        Array::from_shape_vec(shape, data)
    }

    pub fn random_sample(&mut self, shape: &[usize]) -> Array {
        self.random(shape)
    }

    pub fn rand(&mut self, shape: &[usize]) -> Array {
        self.random(shape)
    }

    pub fn randn(&mut self, shape: &[usize]) -> Array {
        let size = compute_size(shape);
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.standard_normal());
        }
        Array::from_shape_vec(shape, data)
    }

    pub fn normal(&mut self, mean: f64, std: f64, shape: &[usize]) -> Array {
        let size = compute_size(shape);
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(mean + std * self.standard_normal());
        }
        Array::from_shape_vec(shape, data)
    }

    pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Array {
        assert!(low < high, "uniform() requires low < high");
        let size = compute_size(shape);
        let scale = high - low;
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.rng.f64() * scale + low);
        }
        Array::from_shape_vec(shape, data)
    }

    pub fn randint(&mut self, low: i64, high: i64, shape: &[usize]) -> Array {
        assert!(low < high, "randint() requires low < high");
        let size = compute_size(shape);
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.rng.i64(low..high) as f64);
        }
        Array::from_shape_vec(shape, data)
    }

    pub fn choice(&mut self, values: &Array, size: usize, replace: bool) -> Array {
        assert!(!values.is_empty(), "choice() requires at least one value");

        if replace {
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                let index = self.rng.usize(0..values.len());
                data.push(values.data()[index]);
            }
            return Array::from_shape_vec(&[size], data);
        }

        assert!(
            size <= values.len(),
            "choice(replace=false) requires size <= number of values"
        );

        let mut indices = (0..values.len()).collect::<Vec<_>>();
        self.shuffle(&mut indices);
        let mut data = Vec::with_capacity(size);
        for index in indices.into_iter().take(size) {
            data.push(values.data()[index]);
        }
        Array::from_shape_vec(&[size], data)
    }

    pub fn permutation(&mut self, values: &Array) -> Array {
        let mut data = values.to_vec();
        self.shuffle(&mut data);
        Array::from_shape_vec(values.shape(), data)
    }
}
