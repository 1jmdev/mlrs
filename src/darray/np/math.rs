use crate::darray::Array;

/// Adds two arrays with broadcasting.
pub fn add(left: &Array, right: &Array) -> Array {
    left.add(right)
}

/// Subtracts two arrays with broadcasting.
pub fn subtract(left: &Array, right: &Array) -> Array {
    left.sub_array(right)
}

/// Multiplies two arrays with broadcasting.
pub fn multiply(left: &Array, right: &Array) -> Array {
    left.multiply(right)
}

/// Divides two arrays with broadcasting.
pub fn divide(left: &Array, right: &Array) -> Array {
    left.divide(right)
}

/// Returns the remainder of elementwise division.
pub fn modulo(left: &Array, right: &Array) -> Array {
    left.modulo(right)
}

/// Raises every element to an integer power.
pub fn poweri(array: &Array, exponent: i32) -> Array {
    array.powi(exponent)
}

/// Raises every element to a floating-point power.
pub fn power(array: &Array, exponent: f64) -> Array {
    array.powf(exponent)
}

/// Adds a scalar to every element.
pub fn add_scalar(array: &Array, scalar: f64) -> Array {
    array.add_scalar(scalar)
}

/// Multiplies every element by a scalar.
pub fn scale(array: &Array, scalar: f64) -> Array {
    array.scale(scalar)
}

/// Returns the additive inverse of every element.
pub fn neg(array: &Array) -> Array {
    array.neg()
}

/// Returns the elementwise absolute value.
pub fn abs(array: &Array) -> Array {
    array.abs()
}

/// Squares every element.
pub fn square(array: &Array) -> Array {
    array.square()
}

/// Returns the elementwise square root.
pub fn sqrt(array: &Array) -> Array {
    array.sqrt()
}

/// Returns the elementwise exponential.
pub fn exp(array: &Array) -> Array {
    array.exp()
}

/// Returns `exp2(x)` for each element.
pub fn exp2(array: &Array) -> Array {
    array.exp2()
}

/// Returns `exp(x) - 1` for each element.
pub fn expm1(array: &Array) -> Array {
    array.expm1()
}

/// Returns the elementwise natural logarithm.
pub fn log(array: &Array) -> Array {
    array.log()
}

/// Returns the elementwise base-10 logarithm.
pub fn log10(array: &Array) -> Array {
    array.log10()
}

/// Returns the elementwise base-2 logarithm.
pub fn log2(array: &Array) -> Array {
    array.log2()
}

/// Returns `ln(1 + x)` for each element.
pub fn log1p(array: &Array) -> Array {
    array.log1p()
}

/// Returns the elementwise sine.
pub fn sin(array: &Array) -> Array {
    array.sin()
}

/// Returns the elementwise cosine.
pub fn cos(array: &Array) -> Array {
    array.cos()
}

/// Returns the elementwise tangent.
pub fn tan(array: &Array) -> Array {
    array.tan()
}

/// Returns the elementwise arcsine.
pub fn asin(array: &Array) -> Array {
    array.asin()
}

/// Returns the elementwise arccosine.
pub fn acos(array: &Array) -> Array {
    array.acos()
}

/// Returns the elementwise arctangent.
pub fn atan(array: &Array) -> Array {
    array.atan()
}

/// Returns the elementwise hyperbolic sine.
pub fn sinh(array: &Array) -> Array {
    array.sinh()
}

/// Returns the elementwise hyperbolic cosine.
pub fn cosh(array: &Array) -> Array {
    array.cosh()
}

/// Returns the elementwise hyperbolic tangent.
pub fn tanh(array: &Array) -> Array {
    array.tanh()
}

/// Returns the elementwise floor.
pub fn floor(array: &Array) -> Array {
    array.floor()
}

/// Returns the elementwise ceiling.
pub fn ceil(array: &Array) -> Array {
    array.ceil()
}

/// Returns the elementwise rounded value.
pub fn round(array: &Array) -> Array {
    array.round()
}

/// Returns the elementwise truncated value.
pub fn trunc(array: &Array) -> Array {
    array.trunc()
}

/// Returns the sign of every element.
pub fn sign(array: &Array) -> Array {
    array.sign()
}

/// Clamps every element to the provided interval.
pub fn clip(array: &Array, min: f64, max: f64) -> Array {
    array.clip(min, max)
}

/// Returns the reciprocal of every element.
pub fn reciprocal(array: &Array) -> Array {
    array.reciprocal()
}

/// Converts radians to degrees.
pub fn degrees(array: &Array) -> Array {
    array.degrees()
}

/// Converts degrees to radians.
pub fn radians(array: &Array) -> Array {
    array.radians()
}

/// Returns the elementwise minimum.
pub fn minimum(left: &Array, right: &Array) -> Array {
    left.minimum(right)
}

/// Returns the elementwise maximum.
pub fn maximum(left: &Array, right: &Array) -> Array {
    left.maximum(right)
}

/// Compares two arrays for equality.
pub fn equal(left: &Array, right: &Array) -> Array {
    left.equal(right)
}

/// Compares two arrays for inequality.
pub fn not_equal(left: &Array, right: &Array) -> Array {
    left.not_equal(right)
}

/// Returns whether each left element is less than the right element.
pub fn less(left: &Array, right: &Array) -> Array {
    left.less(right)
}

/// Returns whether each left element is less than or equal to the right element.
pub fn less_equal(left: &Array, right: &Array) -> Array {
    left.less_equal(right)
}

/// Returns whether each left element is greater than the right element.
pub fn greater(left: &Array, right: &Array) -> Array {
    left.greater(right)
}

/// Returns whether each left element is greater than or equal to the right element.
pub fn greater_equal(left: &Array, right: &Array) -> Array {
    left.greater_equal(right)
}

/// Selects values from `x` or `y` based on a broadcast condition.
pub fn r#where(condition: &Array, x: &Array, y: &Array) -> Array {
    Array::where_cond(condition, x, y)
}

/// Selects values from `x` or `y` based on a broadcast condition.
pub fn where_(condition: &Array, x: &Array, y: &Array) -> Array {
    Array::where_cond(condition, x, y)
}
