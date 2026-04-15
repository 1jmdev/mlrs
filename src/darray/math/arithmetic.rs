use super::super::Array;
use super::super::utils::{binary_map, binary_map_same_shape_simd, unary_map_simd};
use wide::f64x4;

impl Array {
    pub(super) fn binary_op<F>(&self, other: &Self, op: F) -> Self
    where
        F: FnMut(f64, f64) -> f64,
    {
        binary_map(self, other, op)
    }

    /// Adds two arrays using NumPy-style broadcasting.
    pub fn add(&self, other: &Self) -> Self {
        if self.shape == other.shape {
            return binary_map_same_shape_simd(
                self,
                other,
                |left, right| left + right,
                |left, right| left + right,
            );
        }
        if other.is_scalar() {
            return self.add_scalar(other.item());
        }
        if self.is_scalar() {
            return other.add_scalar(self.item());
        }
        self.binary_op(other, |left, right| left + right)
    }

    /// Subtracts two arrays using NumPy-style broadcasting.
    pub fn sub_array(&self, other: &Self) -> Self {
        if self.shape == other.shape {
            return binary_map_same_shape_simd(
                self,
                other,
                |left, right| left - right,
                |left, right| left - right,
            );
        }
        if other.is_scalar() {
            return self.add_scalar(-other.item());
        }
        self.binary_op(other, |left, right| left - right)
    }

    /// Multiplies two arrays using NumPy-style broadcasting.
    pub fn multiply(&self, other: &Self) -> Self {
        if self.shape == other.shape {
            return binary_map_same_shape_simd(
                self,
                other,
                |left, right| left * right,
                |left, right| left * right,
            );
        }
        if other.is_scalar() {
            return self.scale(other.item());
        }
        if self.is_scalar() {
            return other.scale(self.item());
        }
        self.binary_op(other, |left, right| left * right)
    }

    /// Divides two arrays using NumPy-style broadcasting.
    pub fn divide(&self, other: &Self) -> Self {
        if self.shape == other.shape {
            return binary_map_same_shape_simd(
                self,
                other,
                |left, right| left / right,
                |left, right| left / right,
            );
        }
        self.binary_op(other, |left, right| left / right)
    }

    /// Returns the remainder of elementwise division.
    pub fn modulo(&self, other: &Self) -> Self {
        self.binary_op(other, |left, right| left % right)
    }

    /// Returns the elementwise minimum.
    pub fn minimum(&self, other: &Self) -> Self {
        self.binary_op(other, f64::min)
    }

    /// Returns the elementwise maximum.
    pub fn maximum(&self, other: &Self) -> Self {
        self.binary_op(other, f64::max)
    }

    /// Adds a scalar to every element.
    pub fn add_scalar(&self, scalar: f64) -> Self {
        let scalar_values = f64x4::splat(scalar);
        unary_map_simd(self, |value| value + scalar, |value| value + scalar_values)
    }

    /// Multiplies every element by a scalar.
    pub fn scale(&self, scalar: f64) -> Self {
        let scalar_values = f64x4::splat(scalar);
        unary_map_simd(self, |value| value * scalar, |value| value * scalar_values)
    }

    /// Returns the additive inverse of every element.
    pub fn neg(&self) -> Self {
        unary_map_simd(self, |value| -value, |value| -value)
    }
}
