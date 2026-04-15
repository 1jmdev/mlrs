use super::super::Array;
use super::super::utils::sum_simd;

impl Array {
    /// Sums every element and returns a scalar.
    pub fn sum(&self) -> f64 {
        sum_simd(&self.data)
    }

    /// Multiplies every element and returns a scalar.
    pub fn prod(&self) -> f64 {
        self.data.iter().product()
    }

    /// Returns the mean of all elements.
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            f64::NAN
        } else {
            self.sum() / self.data.len() as f64
        }
    }

    /// Returns the population variance of all elements.
    pub fn var(&self) -> f64 {
        if self.data.is_empty() {
            return f64::NAN;
        }

        let mean = self.mean();
        let squared = self
            .data
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .collect::<Vec<_>>();
        sum_simd(&squared) / self.data.len() as f64
    }

    /// Returns the population standard deviation of all elements.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Returns whether all elements are non-zero.
    pub fn all(&self) -> bool {
        self.data.iter().all(|&value| value != 0.0)
    }

    /// Returns whether any element is non-zero.
    pub fn any(&self) -> bool {
        self.data.iter().any(|&value| value != 0.0)
    }

    /// Returns the minimum element.
    pub fn min(&self) -> f64 {
        assert!(!self.data.is_empty(), "min() requires at least one element");
        self.data
            .iter()
            .copied()
            .fold(f64::INFINITY, |current, value| current.min(value))
    }

    /// Returns the maximum element.
    pub fn max(&self) -> f64 {
        assert!(!self.data.is_empty(), "max() requires at least one element");
        self.data
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |current, value| current.max(value))
    }

    /// Returns the flat index of the minimum element.
    pub fn argmin(&self) -> usize {
        assert!(
            !self.data.is_empty(),
            "argmin() requires at least one element"
        );

        let mut best_index = 0;
        let mut best_value = self.data[0];
        for (index, &value) in self.data.iter().enumerate().skip(1) {
            if value < best_value {
                best_index = index;
                best_value = value;
            }
        }
        best_index
    }

    /// Returns the flat index of the maximum element.
    pub fn argmax(&self) -> usize {
        assert!(
            !self.data.is_empty(),
            "argmax() requires at least one element"
        );

        let mut best_index = 0;
        let mut best_value = self.data[0];
        for (index, &value) in self.data.iter().enumerate().skip(1) {
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }
        best_index
    }
}
