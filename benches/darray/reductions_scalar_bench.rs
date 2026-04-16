use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_reductions_scalar(c: &mut Criterion) {
    let a = Array::from_shape_vec(
        &[2_000_000],
        (0..2_000_000).map(|i| (i as f64) * 0.1 + 1.0).collect(),
    );
    let mut g = c.benchmark_group("darray_reductions_scalar");
    g.bench_function("sum", |b| b.iter(|| a.sum()));
    g.bench_function("mean", |b| b.iter(|| a.mean()));
    g.bench_function("var", |b| b.iter(|| a.var()));
    g.bench_function("min", |b| b.iter(|| a.min()));
    g.bench_function("argmax", |b| b.iter(|| a.argmax()));
    g.finish();
}

criterion_group!(benches, bench_reductions_scalar);
criterion_main!(benches);
