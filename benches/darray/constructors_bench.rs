use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_constructors(c: &mut Criterion) {
    let mut g = c.benchmark_group("darray_constructors");
    g.bench_function("from_shape_vec_1m", |b| {
        b.iter(|| Array::from_shape_vec(&[1000, 1000], (0..1_000_000).map(|i| i as f64).collect()))
    });
    g.bench_function("zeros_1m", |b| {
        b.iter(|| Array::zeros(black_box(&[1000, 1000])))
    });
    g.bench_function("ones_1m", |b| {
        b.iter(|| Array::ones(black_box(&[1000, 1000])))
    });
    g.bench_function("linspace_1m", |b| {
        b.iter(|| Array::linspace(black_box(0.0), black_box(1.0), black_box(1_000_000), true))
    });
    g.finish();
}

criterion_group!(benches, bench_constructors);
criterion_main!(benches);
