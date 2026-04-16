use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::darray::Array;
use mlrs::metrics::{
    AccuracyOptions, ClassificationAverage, ClassificationMetricOptions, ConfusionMatrixNormalize,
    ConfusionMatrixOptions, accuracy_score, accuracy_score_with_options, confusion_matrix,
    confusion_matrix_with_options, f1_score, f1_score_with_options, precision_score,
    precision_score_with_options, recall_score, recall_score_with_options,
};

fn data(samples: usize) -> (Array, Array, Array) {
    let y_true = Array::from_shape_vec(&[samples], (0..samples).map(|i| (i % 4) as f64).collect());
    let y_pred = Array::from_shape_vec(
        &[samples],
        (0..samples).map(|i| ((i + (i % 3)) % 4) as f64).collect(),
    );
    let weights = Array::from_shape_vec(
        &[samples],
        (0..samples).map(|i| 0.5 + (i % 5) as f64 * 0.1).collect(),
    );
    (y_true, y_pred, weights)
}

fn bench_classification_metrics(c: &mut Criterion) {
    let (y_true, y_pred, weights) = data(200_000);
    let labels = Array::from_shape_vec(&[4], vec![0.0, 1.0, 2.0, 3.0]);
    let mut g = c.benchmark_group("metrics_classification");
    g.bench_function("accuracy", |b| {
        b.iter(|| accuracy_score(black_box(&y_true), black_box(&y_pred)).expect("accuracy failed"))
    });
    g.bench_function("accuracy_weighted", |b| {
        let options = AccuracyOptions::new().with_sample_weight(&weights);
        b.iter(|| {
            accuracy_score_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(options.clone()),
            )
            .expect("accuracy weighted failed")
        })
    });
    g.bench_function("precision_macro", |b| {
        let options = ClassificationMetricOptions::new()
            .with_average(ClassificationAverage::Macro)
            .with_labels(&labels);
        b.iter(|| {
            precision_score_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(options.clone()),
            )
            .expect("precision macro failed")
        })
    });
    g.bench_function("recall_weighted", |b| {
        let options = ClassificationMetricOptions::new()
            .with_average(ClassificationAverage::Weighted)
            .with_labels(&labels);
        b.iter(|| {
            recall_score_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(options.clone()),
            )
            .expect("recall weighted failed")
        })
    });
    g.bench_function("f1_micro", |b| {
        let options = ClassificationMetricOptions::new()
            .with_average(ClassificationAverage::Micro)
            .with_labels(&labels);
        b.iter(|| {
            f1_score_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(options.clone()),
            )
            .expect("f1 micro failed")
        })
    });
    g.bench_function("confusion_matrix", |b| {
        b.iter(|| {
            confusion_matrix(black_box(&y_true), black_box(&y_pred)).expect("confusion failed")
        })
    });
    g.bench_function("confusion_matrix_norm_true", |b| {
        let options = ConfusionMatrixOptions::new()
            .with_labels(&labels)
            .with_normalize(ConfusionMatrixNormalize::True);
        b.iter(|| {
            confusion_matrix_with_options(
                black_box(&y_true),
                black_box(&y_pred),
                black_box(options.clone()),
            )
            .expect("confusion true failed")
        })
    });
    g.bench_function("scalar_shortcuts", |b| {
        b.iter(|| {
            let _ =
                precision_score(black_box(&y_true), black_box(&y_pred)).expect("precision failed");
            let _ = recall_score(black_box(&y_true), black_box(&y_pred)).expect("recall failed");
            f1_score(black_box(&y_true), black_box(&y_pred)).expect("f1 failed")
        })
    });
    g.finish();
}

criterion_group!(benches, bench_classification_metrics);
criterion_main!(benches);
