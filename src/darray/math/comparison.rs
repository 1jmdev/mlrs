use super::super::Array;
use super::super::utils::ternary_map;

impl Array {
    fn comparison_op<F>(&self, other: &Self, mut op: F) -> Self
    where
        F: FnMut(f64, f64) -> bool,
    {
        self.binary_op(other, |left, right| if op(left, right) { 1.0 } else { 0.0 })
    }

    /// Compares two arrays for equality.
    pub fn equal(&self, other: &Self) -> Self {
        self.comparison_op(other, |left, right| left == right)
    }

    /// Compares two arrays for inequality.
    pub fn not_equal(&self, other: &Self) -> Self {
        self.comparison_op(other, |left, right| left != right)
    }

    /// Returns whether each left element is less than the right element.
    pub fn less(&self, other: &Self) -> Self {
        self.comparison_op(other, |left, right| left < right)
    }

    /// Returns whether each left element is less than or equal to the right element.
    pub fn less_equal(&self, other: &Self) -> Self {
        self.comparison_op(other, |left, right| left <= right)
    }

    /// Returns whether each left element is greater than the right element.
    pub fn greater(&self, other: &Self) -> Self {
        self.comparison_op(other, |left, right| left > right)
    }

    /// Returns whether each left element is greater than or equal to the right element.
    pub fn greater_equal(&self, other: &Self) -> Self {
        self.comparison_op(other, |left, right| left >= right)
    }

    /// Selects values from `x` or `y` based on a broadcast condition.
    pub fn where_cond(condition: &Self, x: &Self, y: &Self) -> Self {
        if condition.shape == x.shape && x.shape == y.shape {
            let mut data = vec![0.0; condition.len()];
            for (index, value) in data.iter_mut().enumerate() {
                *value = if condition.data[index] != 0.0 {
                    x.data[index]
                } else {
                    y.data[index]
                };
            }

            return Self::from_shape_vec(&condition.shape, data);
        }

        ternary_map(
            condition,
            x,
            y,
            |cond, left, right| {
                if cond != 0.0 { left } else { right }
            },
        )
    }
}
