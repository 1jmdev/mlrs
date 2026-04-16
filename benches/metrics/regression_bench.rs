use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::metrics::{
    MultiOutput, RegressionMetricOptions, explained_variance_score, mae, max_error,
    mean_absolute_error_with_options, mean_absolute_percentage_error,
    mean_gamma_deviance_with_options, mean_pinball_loss_with_options, mean_poisson_deviance,
    mean_squared_error, mean_squared_log_error, mean_tweedie_deviance_with_options,
    median_absolute_error, mse, r2_score, r2_score_with_options, rmse, root_mean_squared_error,
    root_mean_squared_log_error,
};

fn data(samples: usize, outputs: usize) -> (Array, Array, Array) {
    let len = samples * outputs;
    let y_true = Array::from_shape_vec(
        &[samples, outputs],
        (0..len).map(|i| (i % 97) as f64 * 0.1 + 1.0).collect(),
    );
    let y_pred = Array::from_shape_vec(
        &[samples, outputs],
        (0..len).map(|i| (i % 89) as f64 * 0.1 + 1.2).collect(),
    );
    let sample_weight = Array::from_shape_vec(
        &[samples],
        (0..samples).map(|i| 0.8 + (i % 7) as f64 * 0.05).collect(),
    );
    (y_true, y_pred, sample_weight)
}

fn bench_regression_metrics(c: &mut Criterion) {
    let (y_true, y_pred, sample_weight) = data(100_000, 1);
    let opts_weighted = RegressionMetricOptions::new().with_sample_weight(&sample_weight);
    let opts_raw = RegressionMetricOptions::new().with_multioutput(MultiOutput::RawValues);
    let mut g = c.benchmark_group("metrics_regression");
    g.bench_function("mean_absolute_error", |b| {
        b.iter(|| {
            mean_absolute_error_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(opts_weighted.clone()),
            )
            .expect("mae weighted failed")
        })
    });
    g.bench_function("median_absolute_error", |b| {
        b.iter(|| {
            median_absolute_error(black_box(&y_true), black_box(&y_pred)).expect("median failed")
        })
    });
    g.bench_function("mean_absolute_percentage_error", |b| {
        b.iter(|| {
            mean_absolute_percentage_error(black_box(&y_true), black_box(&y_pred))
                .expect("mape failed")
        })
    });
    g.bench_function("mean_squared_error", |b| {
        b.iter(|| mean_squared_error(black_box(&y_true), black_box(&y_pred)).expect("mse failed"))
    });
    g.bench_function("root_mean_squared_error", |b| {
        b.iter(|| {
            root_mean_squared_error(black_box(&y_true), black_box(&y_pred)).expect("rmse failed")
        })
    });
    g.bench_function("mean_squared_log_error", |b| {
        b.iter(|| {
            mean_squared_log_error(black_box(&y_true), black_box(&y_pred)).expect("msle failed")
        })
    });
    g.bench_function("root_mean_squared_log_error", |b| {
        b.iter(|| {
            root_mean_squared_log_error(black_box(&y_true), black_box(&y_pred))
                .expect("rmsle failed")
        })
    });
    g.bench_function("explained_variance", |b| {
        b.iter(|| {
            explained_variance_score(black_box(&y_true), black_box(&y_pred)).expect("ev failed")
        })
    });
    g.bench_function("r2_score", |b| {
        b.iter(|| r2_score(black_box(&y_true), black_box(&y_pred)).expect("r2 failed"))
    });
    g.bench_function("r2_raw_values", |b| {
        let (yt, yp, _) = data(50_000, 3);
        b.iter(|| {
            r2_score_with_options(black_box(&yt), black_box(&yp), black_box(opts_raw.clone()))
                .expect("r2 raw failed")
        })
    });
    g.bench_function("mean_pinball_loss", |b| {
        b.iter(|| {
            mean_pinball_loss_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(0.8),
                black_box(opts_weighted.clone()),
            )
            .expect("pinball failed")
        })
    });
    g.bench_function("mean_poisson_deviance", |b| {
        b.iter(|| {
            mean_poisson_deviance(black_box(&y_true), black_box(&y_pred)).expect("poisson failed")
        })
    });
    g.bench_function("mean_gamma_deviance", |b| {
        b.iter(|| {
            mean_gamma_deviance_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(opts_weighted.clone()),
            )
            .expect("gamma failed")
        })
    });
    g.bench_function("mean_tweedie_deviance", |b| {
        b.iter(|| {
            mean_tweedie_deviance_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(1.5),
                black_box(opts_weighted.clone()),
            )
            .expect("tweedie failed")
        })
    });
    g.bench_function("aliases_and_max_error", |b| {
        b.iter(|| {
            let _ = mae(black_box(&y_true), black_box(&y_pred)).expect("mae alias failed");
            let _ = mse(black_box(&y_true), black_box(&y_pred)).expect("mse alias failed");
            let _ = rmse(black_box(&y_true), black_box(&y_pred)).expect("rmse alias failed");
            max_error(black_box(&y_true), black_box(&y_pred)).expect("max error failed")
        })
    });
    g.finish();
}

criterion_group!(benches, bench_regression_metrics);
criterion_main!(benches);
