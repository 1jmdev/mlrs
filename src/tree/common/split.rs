use wide::f64x4;

use super::Criterion;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SplitCandidate {
    pub(crate) feature: usize,
    pub(crate) threshold: f64,
    pub(crate) gain: f64,
}

pub(crate) fn impurity(criterion: Criterion, counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    match criterion {
        Criterion::Gini => 1.0 - squared_probability_sum(counts, total),
        Criterion::Entropy => entropy(counts, total),
    }
}

fn squared_probability_sum(counts: &[usize], total: usize) -> f64 {
    let scale = 1.0 / (total as f64 * total as f64);
    let mut lane_sums = f64x4::ZERO;
    let simd_len = counts.len() / 4 * 4;

    for offset in (0..simd_len).step_by(4) {
        let values = f64x4::from([
            counts[offset] as f64,
            counts[offset + 1] as f64,
            counts[offset + 2] as f64,
            counts[offset + 3] as f64,
        ]);
        lane_sums += values * values;
    }

    let partials: [f64; 4] = lane_sums.into();
    let mut total_sum = partials.into_iter().sum::<f64>();
    total_sum += counts[simd_len..]
        .iter()
        .map(|count| {
            let value = *count as f64;
            value * value
        })
        .sum::<f64>();
    total_sum * scale
}

fn entropy(counts: &[usize], total: usize) -> f64 {
    counts
        .iter()
        .filter(|count| **count != 0)
        .map(|count| {
            let probability = *count as f64 / total as f64;
            -probability * probability.ln()
        })
        .sum()
}
