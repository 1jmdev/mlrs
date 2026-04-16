use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_shape_combine(c: &mut Criterion) {
    let a = Array::from_shape_vec(&[512, 512], (0..262_144).map(|i| i as f64).collect());
    let b2 = Array::from_shape_vec(
        &[512, 512],
        (0..262_144).map(|i| (i as f64) * 0.5).collect(),
    );
    let c3 = Array::from_shape_vec(
        &[512, 512],
        (0..262_144).map(|i| (i as f64) * 2.0).collect(),
    );
    let arrays = [&a, &b2, &c3];
    let mut g = c.benchmark_group("darray_shape_combine");
    g.bench_function("concatenate_axis0", |b| {
        b.iter(|| Array::concatenate(black_box(&arrays), black_box(0)))
    });
    g.bench_function("concatenate_axis1", |b| {
        b.iter(|| Array::concatenate(black_box(&arrays), black_box(1)))
    });
    g.bench_function("stack_axis0", |b| {
        b.iter(|| Array::stack(black_box(&arrays), black_box(0)))
    });
    g.bench_function("stack_axis2", |b| {
        b.iter(|| Array::stack(black_box(&arrays), black_box(2)))
    });
    g.finish();
}

criterion_group!(benches, bench_shape_combine);
criterion_main!(benches);
