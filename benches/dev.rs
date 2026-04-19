use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::linear_model::LinearRegression;
use mlrs::metrics::{CrossValidationOptions, cross_val_score};

fn matrix_data(rows: usize, cols: usize) -> Array {
    Array::from_shape_vec(
        &[rows, cols],
        (0..rows * cols)
            .map(|index| (((index % 97) as f64) - 48.0) * 0.125)
            .collect(),
    )
}

fn vector_data(size: usize) -> Array {
    Array::from_shape_vec(
        &[size],
        (0..size)
            .map(|index| (((index * 17) % 113) as f64) * 0.03125)
            .collect(),
    )
}

fn bench_cross_val_score(c: &mut Criterion) {
    let x = matrix_data(10_000, 32);
    let y = vector_data(10_000);
    let estimator = LinearRegression::new().epochs(120).learning_rate(0.01);
    let options = CrossValidationOptions::new()
        .with_cv(5)
        .with_shuffle(true)
        .with_random_state(42);

    c.bench_function("dev/cross_val_score", |b| {
        b.iter(|| {
            cross_val_score(
                black_box(&estimator),
                black_box(&x),
                black_box(&y),
                black_box(options.clone()),
            )
            .expect("cross_val_score failed")
        })
    });
}

fn bench_shape_relayout(c: &mut Criterion) {
    let a = matrix_data(2_048, 2_048);

    c.bench_function("dev/reshape", |b| {
        b.iter(|| black_box(&a).reshape(black_box(&[1_024, 4_096])))
    });
    c.bench_function("dev/flatten", |b| b.iter(|| black_box(&a).flatten()));
    c.bench_function("dev/squeeze", |b| {
        let expanded = a.expand_dims(0);
        b.iter(|| black_box(&expanded).squeeze())
    });
}

fn bench_axis_ops(c: &mut Criterion) {
    let a = matrix_data(3_000, 512);

    c.bench_function("dev/cumsum_axis", |b| b.iter(|| black_box(&a).cumsum_axis(1)));
    c.bench_function("dev/cumprod_axis", |b| {
        b.iter(|| black_box(&a).cumprod_axis(1))
    });
}

fn bench_transpose_and_tile(c: &mut Criterion) {
    let a = matrix_data(1_024, 1_024);

    c.bench_function("dev/transpose", |b| b.iter(|| black_box(&a).transpose()));
    c.bench_function("dev/tile", |b| b.iter(|| black_box(&a).tile(black_box(&[2, 2]))));
}

criterion_group!(
    benches,
    bench_cross_val_score,
    bench_shape_relayout,
    bench_axis_ops,
    bench_transpose_and_tile
);
criterion_main!(benches);
