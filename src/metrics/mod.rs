mod classification;
mod error;
mod model_selection;
mod regression;

pub use classification::{
    AccuracyOptions, ClassificationAverage, ClassificationMetricOptions,
    ClassificationMetricOutput, ConfusionMatrixNormalize, ConfusionMatrixOptions, accuracy_score,
    accuracy_score_with_options, confusion_matrix, confusion_matrix_with_options, f1_score,
    f1_score_with_options, precision_score, precision_score_with_options, recall_score,
    recall_score_with_options,
};
pub use error::MetricsError;
pub use model_selection::{
    CrossValidationOptions, SplitData, SplitSize, SupervisedEstimator, TrainTestSplitOptions,
    cross_val_score, train_test_split,
};
pub use regression::{
    MultiOutput, RegressionMetricOptions, RegressionMetricOutput, explained_variance_score,
    explained_variance_score_with_options, mae, max_error, mean_absolute_error,
    mean_absolute_error_with_options, mean_absolute_percentage_error,
    mean_absolute_percentage_error_with_options, mean_gamma_deviance,
    mean_gamma_deviance_with_options, mean_pinball_loss, mean_pinball_loss_with_options,
    mean_poisson_deviance, mean_poisson_deviance_with_options, mean_squared_error,
    mean_squared_error_with_options, mean_squared_log_error, mean_squared_log_error_with_options,
    mean_tweedie_deviance, mean_tweedie_deviance_with_options, median_absolute_error,
    median_absolute_error_with_options, mse, r2_score, r2_score_with_options, rmse,
    root_mean_squared_error, root_mean_squared_error_with_options, root_mean_squared_log_error,
    root_mean_squared_log_error_with_options,
};
