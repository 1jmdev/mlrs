use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::linear_model::LinearRegression;
use mlrs::metrics::{
    CrossValidationOptions, SplitSize, TrainTestSplitOptions, cross_val_score, train_test_split,
};

fn data(samples: usize, features: usize) -> (Array, Array) {
    let x = Array::from_shape_vec(
        &[samples, features],
        (0..samples * features)
            .map(|i| ((i % 41) as f64) * 0.01)
            .collect(),
    );
    let y = Array::from_shape_vec(
        &[samples],
        (0..samples).map(|i| (i % 17) as f64 + 1.0).collect(),
    );
    (x, y)
}

fn bench_model_selection(c: &mut Criterion) {
    let (x, y) = data(3000, 24);
    let estimator = LinearRegression::new().epochs(150).learning_rate(0.01);
    let split_options = TrainTestSplitOptions::new()
        .with_test_size(SplitSize::Ratio(0.2))
        .with_shuffle(true)
        .with_random_state(42);
    let cv_options = CrossValidationOptions::new()
        .with_cv(5)
        .with_shuffle(true)
        .with_random_state(7);

    let mut g = c.benchmark_group("metrics_model_selection");
    g.bench_function("train_test_split", |b| {
        b.iter(|| {
            train_test_split(
                black_box(&x),
                black_box(&y),
                black_box(split_options.clone()),
            )
            .expect("split failed")
        })
    });
    g.bench_function("cross_val_score", |b| {
        b.iter(|| {
            cross_val_score(
                black_box(&estimator),
                black_box(&x),
                black_box(&y),
                black_box(cv_options.clone()),
            )
            .expect("cross_val_score failed")
        })
    });
    g.finish();
}

criterion_group!(benches, bench_model_selection);
criterion_main!(benches);
