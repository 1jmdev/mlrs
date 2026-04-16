use crate::darray::Array;

/// Controls whether a split size is specified as a ratio or a sample count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitSize {
    Ratio(f64),
    Count(usize),
}

/// Optional arguments for `train_test_split`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainTestSplitOptions {
    pub test_size: Option<SplitSize>,
    pub train_size: Option<SplitSize>,
    pub shuffle: bool,
    pub random_state: Option<u64>,
}

impl Default for TrainTestSplitOptions {
    fn default() -> Self {
        Self {
            test_size: Some(SplitSize::Ratio(0.25)),
            train_size: None,
            shuffle: true,
            random_state: None,
        }
    }
}

impl TrainTestSplitOptions {
    /// Creates default split options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the test split size.
    pub fn with_test_size(mut self, test_size: SplitSize) -> Self {
        self.test_size = Some(test_size);
        self
    }

    /// Sets the train split size.
    pub fn with_train_size(mut self, train_size: SplitSize) -> Self {
        self.train_size = Some(train_size);
        self
    }

    /// Sets whether the split should be shuffled first.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the optional random seed used for shuffling.
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

/// Holds the arrays returned by `train_test_split`.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitData {
    pub x_train: Array,
    pub x_test: Array,
    pub y_train: Array,
    pub y_test: Array,
}

/// Optional arguments for `cross_val_score`.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossValidationOptions {
    pub cv: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
}

impl Default for CrossValidationOptions {
    fn default() -> Self {
        Self {
            cv: 5,
            shuffle: false,
            random_state: None,
        }
    }
}

impl CrossValidationOptions {
    /// Creates default cross-validation options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of folds.
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Sets whether folds should be shuffled first.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the optional random seed used for shuffling.
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}
