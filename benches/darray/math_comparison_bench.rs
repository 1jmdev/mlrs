use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_math_comparison(c: &mut Criterion) {
    let a = Array::from_shape_vec(&[1_000_000], (0..1_000_000).map(|i| i as f64).collect());
    let b2 = Array::from_shape_vec(
        &[1_000_000],
        (0..1_000_000).map(|i| (i as f64) + 0.1).collect(),
    );
    let mut g = c.benchmark_group("darray_math_comparison");
    g.bench_function("equal", |b| b.iter(|| a.equal(black_box(&b2))));
    g.bench_function("less", |b| b.iter(|| a.less(black_box(&b2))));
    g.bench_function("greater_equal", |b| {
        b.iter(|| a.greater_equal(black_box(&b2)))
    });
    g.bench_function("where_cond", |b| {
        b.iter(|| Array::where_cond(black_box(&a.less(&b2)), black_box(&a), black_box(&b2)))
    });
    g.finish();
}

criterion_group!(benches, bench_math_comparison);
criterion_main!(benches);
