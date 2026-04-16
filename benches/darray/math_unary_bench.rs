use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_math_unary(c: &mut Criterion) {
    let a = Array::from_shape_vec(
        &[1_000_000],
        (0..1_000_000).map(|i| (i as f64) * 0.001 + 1.0).collect(),
    );
    let mut g = c.benchmark_group("darray_math_unary");
    g.bench_function("square", |b| b.iter(|| a.square()));
    g.bench_function("sqrt", |b| b.iter(|| a.sqrt()));
    g.bench_function("exp", |b| b.iter(|| a.exp()));
    g.bench_function("log", |b| b.iter(|| a.log()));
    g.bench_function("clip", |b| {
        b.iter(|| a.clip(black_box(100.0), black_box(500.0)))
    });
    g.finish();
}

criterion_group!(benches, bench_math_unary);
criterion_main!(benches);
