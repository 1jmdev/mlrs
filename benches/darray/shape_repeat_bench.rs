use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;

fn bench_shape_repeat(c: &mut Criterion) {
    let v = Array::from_shape_vec(&[500_000], (0..500_000).map(|i| i as f64).collect());
    let m = Array::from_shape_vec(&[1024, 512], (0..524_288).map(|i| i as f64).collect());
    let mut g = c.benchmark_group("darray_shape_repeat");
    g.bench_function("repeat_flat", |b| b.iter(|| v.repeat(black_box(2), None)));
    g.bench_function("repeat_axis1", |b| {
        b.iter(|| m.repeat(black_box(2), Some(black_box(1))))
    });
    g.bench_function("tile_2d", |b| b.iter(|| m.tile(black_box(&[2, 2]))));
    g.bench_function("tile_3d", |b| b.iter(|| m.tile(black_box(&[2, 1, 2]))));
    g.finish();
}

criterion_group!(benches, bench_shape_repeat);
criterion_main!(benches);
