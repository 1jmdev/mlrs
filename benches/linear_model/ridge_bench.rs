use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::linear_model::Ridge;

fn data(samples: usize, features: usize) -> (Array, Array) {
    let x = Array::from_shape_vec(
        &[samples, features],
        (0..samples * features)
            .map(|i| ((i % 19) as f64) * 0.1)
            .collect(),
    );
    let y = Array::from_shape_vec(&[samples], (0..samples).map(|i| (i % 13) as f64).collect());
    (x, y)
}

fn bench_ridge(c: &mut Criterion) {
    let (x, y) = data(2000, 64);
    let mut trained = Ridge::new(1.0);
    trained.fit(&x, &y).expect("ridge fit failed");
    let mut g = c.benchmark_group("linear_model_ridge");
    g.bench_function("fit", |b| {
        b.iter(|| {
            let mut model = Ridge::new(black_box(1.0));
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

criterion_group!(benches, bench_ridge);
criterion_main!(benches);
