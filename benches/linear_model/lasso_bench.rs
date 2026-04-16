use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::linear_model::Lasso;

fn data(samples: usize, features: usize) -> (Array, Array) {
    let x = Array::from_shape_vec(
        &[samples, features],
        (0..samples * features)
            .map(|i| ((i % 23) as f64) * 0.02)
            .collect(),
    );
    let y = Array::from_shape_vec(
        &[samples],
        (0..samples).map(|i| ((i % 17) as f64) * 0.3).collect(),
    );
    (x, y)
}

fn bench_lasso(c: &mut Criterion) {
    let (x, y) = data(1200, 32);
    let mut trained = Lasso::new(0.01).max_iter(200).tol(1e-4);
    trained.fit(&x, &y).expect("lasso fit failed");
    let mut g = c.benchmark_group("linear_model_lasso");
    g.bench_function("fit", |b| {
        b.iter(|| {
            let mut model = Lasso::new(black_box(0.01))
                .max_iter(black_box(200))
                .tol(black_box(1e-4));
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

criterion_group!(benches, bench_lasso);
criterion_main!(benches);
