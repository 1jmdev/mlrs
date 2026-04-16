use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_shape_transform(c: &mut Criterion) {
    let a = Array::from_shape_vec(&[256, 256, 16], (0..1_048_576).map(|i| i as f64).collect());
    let mut g = c.benchmark_group("darray_shape_transform");
    g.bench_function("reshape", |b| {
        b.iter(|| a.reshape(black_box(&[1024, 1024])))
    });
    g.bench_function("flatten", |b| b.iter(|| a.flatten()));
    g.bench_function("transpose", |b| b.iter(|| a.transpose()));
    g.bench_function("permute_axes", |b| {
        b.iter(|| a.permute_axes(black_box(&[1, 2, 0])))
    });
    g.bench_function("moveaxis", |b| {
        b.iter(|| a.moveaxis(black_box(0), black_box(2)))
    });
    g.finish();
}

criterion_group!(benches, bench_shape_transform);
criterion_main!(benches);
