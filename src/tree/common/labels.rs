use crate::darray::Array;

use super::TreeError;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EncodedLabels {
    pub(crate) classes: Vec<f64>,
    pub(crate) encoded: Vec<usize>,
}

pub(crate) fn encode_labels(y: &Array) -> Result<EncodedLabels, TreeError> {
    if !y.is_vector() || y.is_empty() {
        return Err(TreeError::InvalidLabelShape(y.shape().to_vec()));
    }

    let mut classes = y.to_vec();
    classes.sort_by(f64::total_cmp);
    classes.dedup();
    if classes.len() < 2 {
        return Err(TreeError::InvalidClassCount(classes.len()));
    }

    let encoded = y
        .data()
        .iter()
        .map(|value| {
            classes
                .binary_search_by(|candidate| candidate.total_cmp(value))
                .map_err(|_| TreeError::InvalidLabelShape(y.shape().to_vec()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(EncodedLabels { classes, encoded })
}
