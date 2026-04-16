mod confusion;
mod scores;
mod types;
mod validation;

pub use confusion::{confusion_matrix, confusion_matrix_with_options};
pub use scores::{
    accuracy_score, accuracy_score_with_options, f1_score, f1_score_with_options, precision_score,
    precision_score_with_options, recall_score, recall_score_with_options,
};
pub use types::{
    AccuracyOptions, ClassificationAverage, ClassificationMetricOptions,
    ClassificationMetricOutput, ConfusionMatrixNormalize, ConfusionMatrixOptions,
};
