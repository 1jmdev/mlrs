use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_indexing(c: &mut Criterion) {
    let a = Array::from_shape_vec(&[2000, 512], (0..1_024_000).map(|i| i as f64).collect());
    let idx = (0..1000).map(|i| (i * 7) % 2000).collect::<Vec<_>>();
    let mut g = c.benchmark_group("darray_indexing");
    g.bench_function("row", |b| b.iter(|| a.row(black_box(1234))));
    g.bench_function("column", |b| b.iter(|| a.column(black_box(255))));
    g.bench_function("take_axis0_1000", |b| {
        b.iter(|| a.take(black_box(&idx), black_box(0)))
    });
    g.bench_function("slice_axis0", |b| {
        b.iter(|| a.slice_axis(black_box(0), black_box(100), black_box(1800)))
    });
    g.finish();
}

criterion_group!(benches, bench_indexing);
criterion_main!(benches);
