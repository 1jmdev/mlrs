use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::{Array, RandomState};

fn bench_random(c: &mut Criterion) {
    let mut g = c.benchmark_group("darray_random");
    g.bench_function("random_1m", |b| {
        b.iter(|| {
            let mut rng = RandomState::seeded(black_box(42));
            rng.random(black_box(&[1_000_000]))
        })
    });
    g.bench_function("randn_1m", |b| {
        b.iter(|| {
            let mut rng = RandomState::seeded(black_box(42));
            rng.randn(black_box(&[1_000_000]))
        })
    });
    g.bench_function("choice_no_replace", |b| {
        let values = Array::from_shape_vec(&[200_000], (0..200_000).map(|i| i as f64).collect());
        b.iter(|| {
            let mut rng = RandomState::seeded(black_box(7));
            rng.choice(black_box(&values), black_box(50_000), black_box(false))
                .expect("choice failed")
        })
    });
    g.bench_function("permutation", |b| {
        let values = Array::from_shape_vec(&[500_000], (0..500_000).map(|i| i as f64).collect());
        b.iter(|| {
            let mut rng = RandomState::seeded(black_box(11));
            rng.permutation(black_box(&values))
        })
    });
    g.finish();
}

criterion_group!(benches, bench_random);
criterion_main!(benches);
