/// Generates exponent vectors for polynomial feature expansion.
pub(crate) fn generate_powers(
    n_features: usize,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
) -> Vec<Vec<usize>> {
    let mut powers = Vec::new();
    if include_bias {
        powers.push(vec![0; n_features]);
    }
    if degree == 0 {
        return powers;
    }

    for current_degree in 1..=degree {
        let mut current = vec![0; n_features];
        build_degree(
            &mut powers,
            &mut current,
            n_features,
            current_degree,
            0,
            interaction_only,
        );
    }

    powers
}

fn build_degree(
    powers: &mut Vec<Vec<usize>>,
    current: &mut [usize],
    n_features: usize,
    remaining_degree: usize,
    start_feature: usize,
    interaction_only: bool,
) {
    if remaining_degree == 0 {
        powers.push(current.to_vec());
        return;
    }

    for feature in start_feature..n_features {
        current[feature] += 1;
        build_degree(
            powers,
            current,
            n_features,
            remaining_degree - 1,
            if interaction_only {
                feature + 1
            } else {
                feature
            },
            interaction_only,
        );
        current[feature] -= 1;
    }
}
