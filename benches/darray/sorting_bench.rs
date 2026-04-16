use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::{Array, SearchSide};

fn bench_sorting(c: &mut Criterion) {
    let v = Array::from_shape_vec(
        &[1_000_000],
        (0..1_000_000).rev().map(|i| i as f64).collect(),
    );
    let m = Array::from_shape_vec(
        &[1024, 1024],
        (0..1_048_576).rev().map(|i| i as f64).collect(),
    );
    let sorted = v.sort();
    let probe = Array::from_shape_vec(&[100_000], (0..100_000).map(|i| (i * 10) as f64).collect());
    let mut g = c.benchmark_group("darray_sorting");
    g.bench_function("sort", |b| b.iter(|| v.sort()));
    g.bench_function("argsort", |b| b.iter(|| v.argsort()));
    g.bench_function("sort_axis_1", |b| b.iter(|| m.sort_axis(black_box(1))));
    g.bench_function("argsort_axis_0", |b| {
        b.iter(|| m.argsort_axis(black_box(0)))
    });
    g.bench_function("searchsorted_left", |b| {
        b.iter(|| sorted.searchsorted(black_box(&probe), black_box(SearchSide::Left)))
    });
    g.bench_function("unique", |b| b.iter(|| m.unique()));
    g.finish();
}

criterion_group!(benches, bench_sorting);
criterion_main!(benches);
