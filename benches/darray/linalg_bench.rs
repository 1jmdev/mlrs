use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_linalg(c: &mut Criterion) {
    let v1 = Array::from_shape_vec(
        &[200_000],
        (0..200_000).map(|i| (i as f64) * 0.001).collect(),
    );
    let v2 = Array::from_shape_vec(
        &[200_000],
        (0..200_000).map(|i| (i as f64) * 0.002).collect(),
    );
    let m1 = Array::from_shape_vec(
        &[512, 512],
        (0..262_144).map(|i| (i % 97) as f64 * 0.01).collect(),
    );
    let m2 = Array::from_shape_vec(
        &[512, 512],
        (0..262_144).map(|i| (i % 89) as f64 * 0.02).collect(),
    );
    let mut g = c.benchmark_group("darray_linalg");
    g.bench_function("dot_1d", |b| {
        b.iter(|| v1.dot(black_box(&v2)).expect("dot failed"))
    });
    g.bench_function("matmul_512", |b| {
        b.iter(|| m1.matmul(black_box(&m2)).expect("matmul failed"))
    });
    g.bench_function("outer_2k", |b| {
        let a = Array::from_shape_vec(&[2000], (0..2000).map(|i| i as f64).collect());
        let b2 = Array::from_shape_vec(&[2000], (0..2000).map(|i| (i as f64) * 0.1).collect());
        b.iter(|| a.outer(black_box(&b2)).expect("outer failed"))
    });
    g.bench_function("diag_extract", |b| {
        b.iter(|| m1.diag(black_box(0)).expect("diag failed"))
    });
    g.finish();
}

criterion_group!(benches, bench_linalg);
criterion_main!(benches);
