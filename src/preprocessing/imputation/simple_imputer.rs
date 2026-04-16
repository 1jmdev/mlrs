use crate::darray::Array;
use rayon::prelude::*;

use super::super::PreprocessingError;
use super::super::common::ensure_feature_count;

const PAR_THRESHOLD: usize = 16_384;

/// Controls which per-feature statistic `SimpleImputer` learns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputerStrategy {
    Mean,
    Median,
    MostFrequent,
    Constant,
}

/// Replaces missing values with per-feature summary statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleImputer {
    /// Marks which values should be treated as missing.
    pub missing_values: f64,
    /// Controls which statistic is learned for each feature.
    pub strategy: ImputerStrategy,
    /// Provides the replacement used by the constant strategy.
    pub fill_value: Option<f64>,
    /// Stores learned replacement values for each feature.
    pub statistics_: Option<Array>,
    /// Stores the number of features observed during fitting.
    pub n_features_in_: Option<usize>,
}

impl Default for SimpleImputer {
    /// Returns the sklearn-compatible default configuration.
    fn default() -> Self {
        Self {
            missing_values: f64::NAN,
            strategy: ImputerStrategy::Mean,
            fill_value: None,
            statistics_: None,
            n_features_in_: None,
        }
    }
}

impl SimpleImputer {
    /// Creates a simple imputer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy with a specific missing-value marker.
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Returns a copy with a specific imputation strategy.
    pub fn strategy(mut self, strategy: ImputerStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Returns a copy with a specific constant replacement value.
    pub fn fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = Some(fill_value);
        self
    }

    /// Reports whether learned statistics are available.
    pub fn is_fitted(&self) -> bool {
        self.statistics_.is_some() && self.n_features_in_.is_some()
    }

    /// Returns the fitted replacement statistic for each feature.
    pub fn statistics(&self) -> Result<&Array, PreprocessingError> {
        self.statistics_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("SimpleImputer"))
    }

    /// Learns per-feature replacement values from a feature matrix.
    pub fn fit(&mut self, x: &Array) -> Result<&mut Self, PreprocessingError> {
        let (rows, cols) = ensure_2d_imputer_input(x, self.missing_values, "X")?;
        let statistics = match self.strategy {
            ImputerStrategy::Mean => fit_mean_statistics(x, rows, cols, self.missing_values)?,
            ImputerStrategy::Median => fit_median_statistics(x, rows, cols, self.missing_values)?,
            ImputerStrategy::MostFrequent => {
                fit_most_frequent_statistics(x, rows, cols, self.missing_values)?
            }
            ImputerStrategy::Constant => vec![checked_fill_value(self.fill_value)?; cols],
        };

        self.statistics_ = Some(Array::from_shape_vec(&[cols], statistics));
        self.n_features_in_ = Some(cols);
        Ok(self)
    }

    /// Fits the imputer and returns the transformed feature matrix.
    pub fn fit_transform(&mut self, x: &Array) -> Result<Array, PreprocessingError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Replaces missing values using the fitted per-feature statistics.
    pub fn transform(&self, x: &Array) -> Result<Array, PreprocessingError> {
        let (rows, cols) = ensure_2d_imputer_input(x, self.missing_values, "X")?;
        ensure_feature_count(
            cols,
            self.n_features_in_
                .ok_or(PreprocessingError::NotFitted("SimpleImputer"))?,
        )?;

        let statistics = self
            .statistics_
            .as_ref()
            .ok_or(PreprocessingError::NotFitted("SimpleImputer"))?
            .data();
        let mut data = vec![0.0; rows * cols];
        if data.len() >= PAR_THRESHOLD {
            data.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, output)| {
                    let input = &x.data()[row * cols..(row + 1) * cols];
                    for col in 0..cols {
                        let value = input[col];
                        output[col] = if is_missing(value, self.missing_values) {
                            statistics[col]
                        } else {
                            value
                        };
                    }
                });
        } else {
            for row in 0..rows {
                let input = &x.data()[row * cols..(row + 1) * cols];
                let output = &mut data[row * cols..(row + 1) * cols];
                for col in 0..cols {
                    let value = input[col];
                    output[col] = if is_missing(value, self.missing_values) {
                        statistics[col]
                    } else {
                        value
                    };
                }
            }
        }

        Ok(Array::from_shape_vec(&[rows, cols], data))
    }
}

fn ensure_2d_imputer_input(
    x: &Array,
    missing_values: f64,
    name: &'static str,
) -> Result<(usize, usize), PreprocessingError> {
    let shape = x.shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PreprocessingError::InvalidInputShape(shape.to_vec()));
    }
    if x.data().iter().any(|&value| {
        value.is_infinite() || (!value.is_finite() && !is_missing(value, missing_values))
    }) {
        return Err(PreprocessingError::NonFiniteInput(name));
    }
    Ok((shape[0], shape[1]))
}

fn checked_fill_value(fill_value: Option<f64>) -> Result<f64, PreprocessingError> {
    let value = fill_value.unwrap_or(0.0);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PreprocessingError::InvalidFillValue(value))
    }
}

fn fit_mean_statistics(
    x: &Array,
    rows: usize,
    cols: usize,
    missing_values: f64,
) -> Result<Vec<f64>, PreprocessingError> {
    let compute = |col: usize| {
        let (sum, count) = (0..rows).fold((0.0, 0usize), |(sum, count), row| {
            let value = x.data()[row * cols + col];
            if is_missing(value, missing_values) {
                (sum, count)
            } else {
                (sum + value, count + 1)
            }
        });
        if count == 0 {
            Err(PreprocessingError::MissingStatistic {
                feature_index: col,
                strategy: "mean",
            })
        } else {
            Ok(sum / count as f64)
        }
    };

    if rows * cols >= PAR_THRESHOLD {
        (0..cols).into_par_iter().map(compute).collect()
    } else {
        (0..cols).map(compute).collect()
    }
}

fn fit_median_statistics(
    x: &Array,
    rows: usize,
    cols: usize,
    missing_values: f64,
) -> Result<Vec<f64>, PreprocessingError> {
    let compute = |col: usize| {
        let mut values = (0..rows)
            .map(|row| x.data()[row * cols + col])
            .filter(|&value| !is_missing(value, missing_values))
            .collect::<Vec<_>>();
        if values.is_empty() {
            return Err(PreprocessingError::MissingStatistic {
                feature_index: col,
                strategy: "median",
            });
        }
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        Ok(if values.len() % 2 == 0 {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        })
    };

    if rows * cols >= PAR_THRESHOLD {
        (0..cols).into_par_iter().map(compute).collect()
    } else {
        (0..cols).map(compute).collect()
    }
}

fn fit_most_frequent_statistics(
    x: &Array,
    rows: usize,
    cols: usize,
    missing_values: f64,
) -> Result<Vec<f64>, PreprocessingError> {
    let compute = |col: usize| {
        let mut values = (0..rows)
            .map(|row| x.data()[row * cols + col])
            .filter(|&value| !is_missing(value, missing_values))
            .collect::<Vec<_>>();
        if values.is_empty() {
            return Err(PreprocessingError::MissingStatistic {
                feature_index: col,
                strategy: "most_frequent",
            });
        }
        values.sort_by(f64::total_cmp);

        let mut best_value = values[0];
        let mut best_count = 1usize;
        let mut current_value = values[0];
        let mut current_count = 1usize;

        for &value in values.iter().skip(1) {
            if value.total_cmp(&current_value).is_eq() {
                current_count += 1;
            } else {
                if current_count > best_count {
                    best_value = current_value;
                    best_count = current_count;
                }
                current_value = value;
                current_count = 1;
            }
        }
        if current_count > best_count {
            best_value = current_value;
        }

        Ok(best_value)
    };

    if rows * cols >= PAR_THRESHOLD {
        (0..cols).into_par_iter().map(compute).collect()
    } else {
        (0..cols).map(compute).collect()
    }
}

fn is_missing(value: f64, missing_values: f64) -> bool {
    if missing_values.is_nan() {
        value.is_nan()
    } else {
        value.total_cmp(&missing_values).is_eq()
    }
}
