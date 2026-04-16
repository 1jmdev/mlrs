use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::linear_model::LinearRegression;

fn data(samples: usize, features: usize) -> (Array, Array) {
    let x = Array::from_shape_vec(
        &[samples, features],
        (0..samples * features)
            .map(|i| ((i % 37) as f64) * 0.05)
            .collect(),
    );
    let y = Array::from_shape_vec(
        &[samples],
        (0..samples)
            .map(|i| (i % 7) as f64 + (i % 11) as f64 * 0.2)
            .collect(),
    );
    (x, y)
}

fn bench_linear_regression(c: &mut Criterion) {
    let (x, y) = data(1500, 32);
    let mut trained = LinearRegression::new().epochs(200).learning_rate(0.01);
    trained.fit(&x, &y).expect("linear regression fit failed");
    let mut g = c.benchmark_group("linear_model_linear_regression");
    g.bench_function("fit", |b| {
        b.iter(|| {
            let mut model = LinearRegression::new()
                .epochs(black_box(200))
                .learning_rate(black_box(0.01));
            let _ = model.fit(black_box(&x), black_box(&y)).expect("fit failed");
        })
    });
    g.bench_function("predict", |b| {
        b.iter(|| trained.predict(black_box(&x)).expect("predict failed"))
    });
    g.bench_function("score", |b| {
        b.iter(|| {
            trained
                .score(black_box(&x), black_box(&y))
                .expect("score failed")
        })
    });
    g.finish();
}

criterion_group!(benches, bench_linear_regression);
criterion_main!(benches);
