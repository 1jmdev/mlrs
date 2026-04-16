mod simd;
mod statistics;
mod validation;

pub(crate) use simd::{SIMD_LANES, load_f64x4, reduce_max, reduce_sum, store_f64x4};
pub(crate) use statistics::{
    column_mean_var, column_min_max, column_percentiles, unique_sorted, unique_sorted_1d,
};
pub(crate) use validation::{
    checked_feature_range, checked_quantile_range, ensure_1d_finite, ensure_2d_finite,
    ensure_feature_count, is_effectively_zero,
};
