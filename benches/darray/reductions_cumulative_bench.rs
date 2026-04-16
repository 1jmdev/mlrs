use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_reductions_cumulative(c: &mut Criterion) {
    let a = Array::from_shape_vec(
        &[1_000_000],
        (0..1_000_000).map(|i| (i % 11) as f64 + 1.0).collect(),
    );
    let m = Array::from_shape_vec(
        &[1024, 1024],
        (0..1_048_576).map(|i| (i % 13) as f64 + 1.0).collect(),
    );
    let mut g = c.benchmark_group("darray_reductions_cumulative");
    g.bench_function("cumsum", |b| b.iter(|| a.cumsum()));
    g.bench_function("cumprod", |b| b.iter(|| a.cumprod()));
    g.bench_function("cumsum_axis_0", |b| b.iter(|| m.cumsum_axis(black_box(0))));
    g.bench_function("cumprod_axis_1", |b| {
        b.iter(|| m.cumprod_axis(black_box(1)))
    });
    g.finish();
}

criterion_group!(benches, bench_reductions_cumulative);
criterion_main!(benches);
