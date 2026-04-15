use smallvec::SmallVec;

pub(crate) type ShapeVec = SmallVec<[usize; 6]>;

pub(crate) const PAR_THRESHOLD: usize = 16_384;
pub(crate) const PAR_CHUNK_LEN: usize = 4_096;

pub(crate) fn compute_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

pub(crate) fn compute_strides(shape: &[usize]) -> ShapeVec {
    if shape.is_empty() {
        return ShapeVec::new();
    }

    let mut strides = ShapeVec::from_elem(1, shape.len());
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

pub(crate) fn validate_shape_data(shape: &[usize], len: usize) {
    let expected = compute_size(shape);
    assert_eq!(
        expected, len,
        "shape {:?} expects {} elements, got {}",
        shape, expected, len
    );
}

pub(crate) fn normalize_axis(axis: usize, ndim: usize) {
    assert!(axis < ndim, "axis {axis} out of bounds for {ndim}-D array");
}

pub(crate) fn index_to_offset(indices: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    assert_eq!(
        indices.len(),
        shape.len(),
        "index {:?} does not match {} dimensions",
        indices,
        shape.len()
    );

    indices
        .iter()
        .zip(shape.iter().zip(strides.iter()))
        .map(|(&index, (&dimension, &stride))| {
            assert!(
                index < dimension,
                "index {index} out of bounds for dimension {dimension}"
            );
            index * stride
        })
        .sum()
}

pub(crate) fn reduced_shape(shape: &[usize], axis: usize) -> ShapeVec {
    shape
        .iter()
        .enumerate()
        .filter_map(|(current_axis, &dimension)| (current_axis != axis).then_some(dimension))
        .collect::<ShapeVec>()
}

pub(crate) fn infer_shape(new_shape: &[isize], total: usize) -> ShapeVec {
    let inferred_count = new_shape
        .iter()
        .filter(|&&dimension| dimension == -1)
        .count();
    assert!(
        inferred_count <= 1,
        "only one inferred dimension is allowed, got {:?}",
        new_shape
    );

    let known_product = new_shape
        .iter()
        .filter(|&&dimension| dimension != -1)
        .map(|&dimension| {
            assert!(
                dimension > 0,
                "shape dimensions must be positive or -1, got {dimension}"
            );
            dimension as usize
        })
        .product::<usize>();

    new_shape
        .iter()
        .map(|&dimension| {
            if dimension == -1 {
                assert!(
                    known_product != 0 && total % known_product == 0,
                    "cannot infer shape {:?} for {} elements",
                    new_shape,
                    total
                );
                total / known_product
            } else {
                dimension as usize
            }
        })
        .collect::<ShapeVec>()
}

#[inline]
pub(crate) fn axis_inner_outer(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    let inner = shape[axis + 1..].iter().product::<usize>();
    let outer = if axis == 0 {
        1
    } else {
        shape[..axis].iter().product::<usize>()
    };
    (inner, outer, shape[axis])
}
