use super::layout::ShapeVec;

pub(crate) fn broadcast_shape(left: &[usize], right: &[usize]) -> ShapeVec {
    let ndim = left.len().max(right.len());
    let mut shape = ShapeVec::from_elem(1, ndim);

    for index in 0..ndim {
        let left_dimension = *left.get(left.len().wrapping_sub(index + 1)).unwrap_or(&1);
        let right_dimension = *right.get(right.len().wrapping_sub(index + 1)).unwrap_or(&1);

        assert!(
            left_dimension == right_dimension || left_dimension == 1 || right_dimension == 1,
            "cannot broadcast shapes {:?} and {:?}",
            left,
            right
        );

        shape[ndim - index - 1] = left_dimension.max(right_dimension);
    }

    shape
}

pub(crate) fn broadcast_shapes(shapes: &[&[usize]]) -> ShapeVec {
    let mut result = ShapeVec::new();
    for shape in shapes {
        result = broadcast_shape(&result, shape);
    }
    result
}

pub(crate) fn broadcast_strides(
    source_shape: &[usize],
    source_strides: &[usize],
    target_shape: &[usize],
) -> ShapeVec {
    let padding = target_shape.len().saturating_sub(source_shape.len());
    let mut result = ShapeVec::with_capacity(target_shape.len());

    for (axis, &target_dimension) in target_shape.iter().enumerate() {
        if axis < padding {
            result.push(0);
            continue;
        }

        let source_axis = axis - padding;
        let source_dimension = source_shape[source_axis];
        let source_stride = source_strides[source_axis];

        assert!(
            source_dimension == target_dimension || source_dimension == 1,
            "cannot broadcast shape {:?} into {:?}",
            source_shape,
            target_shape
        );

        if source_dimension == 1 && target_dimension != 1 {
            result.push(0);
        } else {
            result.push(source_stride);
        }
    }

    result
}

pub(crate) fn offsets_for_broadcast(
    target_shape: &[usize],
    target_strides: &[usize],
    source_broadcast_strides: &[usize],
    flat_index: usize,
) -> usize {
    if target_shape.is_empty() {
        return 0;
    }

    let mut remainder = flat_index;
    let mut offset = 0;
    for axis in 0..target_shape.len() {
        let coordinate = remainder / target_strides[axis];
        remainder %= target_strides[axis];
        offset += coordinate * source_broadcast_strides[axis];
    }
    offset
}
