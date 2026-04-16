use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::linear_model::LogisticRegression;

fn data(samples: usize, features: usize) -> (Array, Array) {
    let x = Array::from_shape_vec(
        &[samples, features],
        (0..samples * features)
            .map(|i| ((i % 31) as f64) * 0.05)
            .collect(),
    );
    let y = Array::from_shape_vec(&[samples], (0..samples).map(|i| (i % 3) as f64).collect());
    (x, y)
}

fn bench_logistic_regression(c: &mut Criterion) {
    let (x, y) = data(1500, 24);
    let mut trained = LogisticRegression::new()
        .max_iter(150)
        .learning_rate(0.05)
        .tol(1e-4);
    trained.fit(&x, &y).expect("logistic regression fit failed");
    let mut g = c.benchmark_group("linear_model_logistic_regression");
    g.bench_function("fit", |b| {
        b.iter(|| {
            let mut model = LogisticRegression::new()
                .max_iter(black_box(150))
                .learning_rate(black_box(0.05))
                .tol(black_box(1e-4));
            let _ = model.fit(black_box(&x), black_box(&y)).expect("fit failed");
        })
    });
    g.bench_function("predict_proba", |b| {
        b.iter(|| {
            trained
                .predict_proba(black_box(&x))
                .expect("predict_proba failed")
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

criterion_group!(benches, bench_logistic_regression);
criterion_main!(benches);
