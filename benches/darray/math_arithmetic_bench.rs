use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_math_arithmetic(c: &mut Criterion) {
    let a = Array::from_shape_vec(&[1_000_000], (0..1_000_000).map(|i| i as f64).collect());
    let b2 = Array::from_shape_vec(
        &[1_000_000],
        (0..1_000_000).map(|i| (i as f64) * 0.5 + 1.0).collect(),
    );
    let mut g = c.benchmark_group("darray_math_arithmetic");
    g.bench_function("add", |b| b.iter(|| a.add(black_box(&b2))));
    g.bench_function("sub_array", |b| b.iter(|| a.sub_array(black_box(&b2))));
    g.bench_function("multiply", |b| b.iter(|| a.multiply(black_box(&b2))));
    g.bench_function("divide", |b| b.iter(|| a.divide(black_box(&b2))));
    g.bench_function("scale", |b| b.iter(|| a.scale(black_box(1.5))));
    g.finish();
}

criterion_group!(benches, bench_math_arithmetic);
criterion_main!(benches);
