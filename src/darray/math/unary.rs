use super::super::Array;
use super::super::utils::{unary_map, unary_map_simd};
use wide::f64x4;

impl Array {
    /// Raises each element to an integer power.
    pub fn powi(&self, exponent: i32) -> Self {
        unary_map(self, |value| value.powi(exponent))
    }

    /// Raises each element to a floating-point power.
    pub fn powf(&self, exponent: f64) -> Self {
        unary_map(self, |value| value.powf(exponent))
    }

    /// Squares every element.
    pub fn square(&self) -> Self {
        unary_map_simd(self, |value| value * value, |value| value * value)
    }

    /// Returns the elementwise absolute value.
    pub fn abs(&self) -> Self {
        unary_map(self, f64::abs)
    }

    /// Returns the elementwise square root.
    pub fn sqrt(&self) -> Self {
        unary_map(self, f64::sqrt)
    }

    /// Returns the elementwise exponential.
    pub fn exp(&self) -> Self {
        unary_map(self, f64::exp)
    }

    /// Returns the elementwise natural logarithm.
    pub fn log(&self) -> Self {
        unary_map(self, f64::ln)
    }

    /// Returns the elementwise base-10 logarithm.
    pub fn log10(&self) -> Self {
        unary_map(self, f64::log10)
    }

    /// Returns the elementwise base-2 logarithm.
    pub fn log2(&self) -> Self {
        unary_map(self, f64::log2)
    }

    /// Returns `exp2(x)` for each element.
    pub fn exp2(&self) -> Self {
        unary_map(self, f64::exp2)
    }

    /// Returns `exp(x) - 1` for each element.
    pub fn expm1(&self) -> Self {
        unary_map(self, f64::exp_m1)
    }

    /// Returns `ln(1 + x)` for each element.
    pub fn log1p(&self) -> Self {
        unary_map(self, f64::ln_1p)
    }

    /// Returns the elementwise sine.
    pub fn sin(&self) -> Self {
        unary_map(self, f64::sin)
    }

    /// Returns the elementwise cosine.
    pub fn cos(&self) -> Self {
        unary_map(self, f64::cos)
    }

    /// Returns the elementwise tangent.
    pub fn tan(&self) -> Self {
        unary_map(self, f64::tan)
    }

    /// Returns the elementwise arcsine.
    pub fn asin(&self) -> Self {
        unary_map(self, f64::asin)
    }

    /// Returns the elementwise arccosine.
    pub fn acos(&self) -> Self {
        unary_map(self, f64::acos)
    }

    /// Returns the elementwise arctangent.
    pub fn atan(&self) -> Self {
        unary_map(self, f64::atan)
    }

    /// Returns the elementwise hyperbolic sine.
    pub fn sinh(&self) -> Self {
        unary_map(self, f64::sinh)
    }

    /// Returns the elementwise hyperbolic cosine.
    pub fn cosh(&self) -> Self {
        unary_map(self, f64::cosh)
    }

    /// Returns the elementwise hyperbolic tangent.
    pub fn tanh(&self) -> Self {
        unary_map(self, f64::tanh)
    }

    /// Returns the elementwise floor.
    pub fn floor(&self) -> Self {
        unary_map(self, f64::floor)
    }

    /// Returns the elementwise ceiling.
    pub fn ceil(&self) -> Self {
        unary_map(self, f64::ceil)
    }

    /// Returns the elementwise rounded value.
    pub fn round(&self) -> Self {
        unary_map(self, f64::round)
    }

    /// Returns the elementwise truncated value.
    pub fn trunc(&self) -> Self {
        unary_map(self, f64::trunc)
    }

    /// Returns the sign of every element.
    pub fn sign(&self) -> Self {
        unary_map(self, |value| {
            if value > 0.0 {
                1.0
            } else if value < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    /// Converts radians to degrees.
    pub fn degrees(&self) -> Self {
        unary_map(self, f64::to_degrees)
    }

    /// Converts degrees to radians.
    pub fn radians(&self) -> Self {
        unary_map(self, f64::to_radians)
    }

    /// Clamps every element to the provided interval.
    pub fn clip(&self, min: f64, max: f64) -> Self {
        unary_map(self, |value| value.clamp(min, max))
    }

    /// Returns the reciprocal of every element.
    pub fn reciprocal(&self) -> Self {
        let ones = f64x4::splat(1.0);
        unary_map_simd(self, |value| 1.0 / value, |value| ones / value)
    }
}
