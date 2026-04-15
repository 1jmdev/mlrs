# MLRS

A from-scratch machine learning library written in Rust. No external ML frameworks -- just raw linear algebra, SIMD acceleration, and parallel computation built on top of a custom n-dimensional array engine.

## Overview

mlrs is built around two core modules:

- **darray** -- A dense, row-major, n-dimensional array of `f64` values with broadcasting, SIMD-accelerated math, and Rayon-based parallelism.
- **linear_model** -- Gradient-descent-based linear regression supporting single and multi-target fitting, R² scoring, and residual computation.

## Array Engine

The `darray` module provides a `Array` type backed by flat `Vec<f64>` storage with precomputed shape and strides. It exposes a functional-style API through the `np` namespace module.

### Construction

```rust
use mlrs::darray::np;

let a = np::array(&[1.0, 2.0, 3.0]);
let b = np::zeros(&[3, 3]);
let c = np::ones(&[2, 4]);
let d = np::eye(3, 3, 0);
let e = np::arange(0.0, 10.0, 1.0);
let f = np::linspace(0.0, 1.0, 50, true);
let g = np::full(&[2, 2], 7.0);
let id = np::identity(4);
```

### Arithmetic & Broadcasting

All binary operations support full broadcasting:

```rust
let a = np::array(&[1.0, 2.0, 3.0]).reshape(&[3, 1]);
let b = np::array(&[10.0, 20.0]);

let sum = a.add(&b);        // [3, 2]
let diff = a.sub_array(&b);
let prod = a.multiply(&b);
let quot = a.divide(&b);
let rem = a.modulo(&b);
```

Scalar operations:

```rust
let scaled = a.scale(2.0);
let shifted = a.add_scalar(1.0);
let negated = a.neg();
```

### Math Functions

Unary element-wise operations:

```rust
a.sqrt();  a.exp();   a.log();    a.log2();   a.log10();
a.sin();   a.cos();   a.tan();    a.asin();   a.acos();   a.atan();
a.sinh();  a.cosh();  a.tanh();
a.abs();   a.square(); a.reciprocal();
a.floor(); a.ceil();  a.round();  a.trunc();  a.sign();
a.clip(0.0, 1.0);
a.degrees(); a.radians();
a.powi(3); a.powf(0.5);
a.exp2();  a.expm1(); a.log1p();
```

### Comparisons & Conditionals

```rust
let mask = a.greater(&b);
let result = Array::where_cond(&mask, &x, &y);

a.equal(&b);  a.not_equal(&b);
a.less(&b);   a.less_equal(&b);
a.greater(&b); a.greater_equal(&b);
```

### Reductions

Global reductions return scalars, axis reductions return arrays:

```rust
a.sum();    a.sum_axis(0);
a.prod();   a.prod_axis(1);
a.mean();   a.mean_axis(0);
a.var();    a.var_axis(1);
a.std();    a.std_axis(0);
a.min();    a.min_axis(0);
a.max();    a.max_axis(1);
a.argmin(); a.argmin_axis(0);
a.argmax(); a.argmax_axis(1);
a.all();    a.all_axis(0);
a.any();    a.any_axis(1);
a.cumsum(); a.cumsum_axis(0);
a.cumprod(); a.cumprod_axis(1);
```

### Shape Manipulation

```rust
a.reshape(&[2, -1]);        // infer one dimension with -1
a.flatten();
a.transpose();
a.expand_dims(0);
a.squeeze();
a.swapaxes(0, 1);
a.moveaxis(0, 2);
a.permute_axes(&[2, 0, 1]);

Array::concatenate(&[&a, &b], 0);
Array::stack(&[&a, &b], 0);
a.repeat(3, Some(1));
a.tile(&[2, 3]);
```

### Indexing

```rust
a.get(&[0, 1]);
a.set(&[0, 1], 5.0);
a.item();                   // scalar from 0-D or single-element array
a.row(0);
a.column(1);
a.take(&[0, 2], 1);
a.slice_axis(0, 1, 3);
```

### Linear Algebra

```rust
a.dot(&b);                  // 1-D inner, 2-D x 1-D, or 2-D x 2-D
a.matmul(&b);               // matrix multiplication (uses matrixmultiply crate)
a.outer(&b);                // vector outer product
a.vdot(&b);                 // flattened dot product
a.diag(0);                  // extract or construct diagonal
a.diagflat(0);
a.trace(0);
```

Matrix multiplication delegates to the `matrixmultiply` crate for cache-efficient `dgemm`.

### Sorting & Search

```rust
a.sort();                   // sorted flattened copy
a.argsort();
a.sort_axis(1);
a.argsort_axis(0);
a.searchsorted(&values, SearchSide::Left);

a.nonzero();                // per-axis indices of non-zero elements
a.flatnonzero();
a.argwhere();
a.unique();
```

### Random

```rust
use mlrs::darray::np::random;

let mut rng = random::seeded(42);
rng.random(&[3, 3]);        // uniform [0, 1)
rng.randn(&[100]);          // standard normal (Box-Muller)
rng.normal(0.0, 1.0, &[5]);
rng.uniform(-1.0, 1.0, &[10]);
rng.randint(0, 10, &[4]);
rng.choice(&pool, 5, false);
rng.permutation(&arr);
```

Convenience functions that create a fresh RNG per call are also available:

```rust
let x = random::random(&[3, 3]);
let y = random::randn(&[100]);
```

## Linear Regression

Gradient-descent-based linear regression with SIMD-accelerated training:

```rust
use mlrs::{linear_model::LinearRegression, darray::np};

let x = np::array(&[1.0, 2.0, 3.0, 4.0, 5.0]).reshape(&[-1, 1]);
let y = np::array(&[2.0, 4.0, 6.0, 8.0, 10.0]);

let mut model = LinearRegression::new()
    .epochs(1000)
    .learning_rate(0.01)
    .fit_intercept(true);

model.fit(&x, &y)?;

let coef = model.coef()?;
let prediction = model.predict(&np::array(&[7.0]).reshape(&[-1, 1]))?;
let r2 = model.score(&x, &y)?;
```

Multi-target regression is supported by passing a 2-D `y` matrix.

## Performance

- **SIMD**: Arithmetic, reductions, dot products, and training loops use 4-wide `f64x4` vectorization via the `wide` crate.
- **Parallelism**: Operations on arrays larger than 16,384 elements automatically dispatch to Rayon thread pools.
- **Matrix multiply**: 2-D `matmul` and gradient descent use `matrixmultiply::dgemm` for cache-friendly blocked multiplication.
- **Small shapes**: Shape and stride vectors use `SmallVec<[usize; 6]>` to avoid heap allocation for arrays up to 6 dimensions.

## Dependencies

| Crate | Purpose |
|---|---|
| `fastrand` | Lightweight PRNG for random array generation |
| `matrixmultiply` | Cache-efficient `dgemm` for matrix multiplication |
| `memchr` | Byte-level search utilities |
| `rayon` | Data-parallel iteration for large arrays |
| `smallvec` | Stack-allocated shape/stride vectors |
| `wide` | Portable SIMD (`f64x4`) |

## Building

```sh
cargo build --release
```

## License

[MIT](LICENSE)
