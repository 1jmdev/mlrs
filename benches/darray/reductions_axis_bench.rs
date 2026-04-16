use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_reductions_axis(c: &mut Criterion) {
    let a = Array::from_shape_vec(
        &[1024, 1024],
        (0..1_048_576).map(|i| (i % 97) as f64 + 1.0).collect(),
    );
    let mut g = c.benchmark_group("darray_reductions_axis");
    g.bench_function("sum_axis_0", |b| b.iter(|| a.sum_axis(black_box(0))));
    g.bench_function("mean_axis_1", |b| b.iter(|| a.mean_axis(black_box(1))));
    g.bench_function("var_axis_1", |b| b.iter(|| a.var_axis(black_box(1))));
    g.bench_function("argmin_axis_0", |b| b.iter(|| a.argmin_axis(black_box(0))));
    g.bench_function("argmax_axis_1", |b| b.iter(|| a.argmax_axis(black_box(1))));
    g.finish();
}

criterion_group!(benches, bench_reductions_axis);
criterion_main!(benches);
