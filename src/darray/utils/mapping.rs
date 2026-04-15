use super::super::Array;
use super::broadcast::{
    broadcast_shape, broadcast_shapes, broadcast_strides, offsets_for_broadcast,
};
use super::layout::{compute_size, compute_strides, normalize_axis, reduced_shape};

pub(crate) fn unary_map<F>(array: &Array, mut op: F) -> Array
where
    F: FnMut(f64) -> f64,
{
    Array {
        data: array.data.iter().copied().map(&mut op).collect(),
        shape: array.shape.clone(),
        strides: array.strides.clone(),
    }
}

pub(crate) fn binary_map<F>(left: &Array, right: &Array, mut op: F) -> Array
where
    F: FnMut(f64, f64) -> f64,
{
    let shape = broadcast_shape(&left.shape, &right.shape);
    let strides = compute_strides(&shape);
    let left_strides = broadcast_strides(&left.shape, &left.strides, &shape);
    let right_strides = broadcast_strides(&right.shape, &right.strides, &shape);
    let size = compute_size(&shape);

    let data = (0..size)
        .map(|flat_index| {
            let left_offset = offsets_for_broadcast(&shape, &strides, &left_strides, flat_index);
            let right_offset = offsets_for_broadcast(&shape, &strides, &right_strides, flat_index);
            op(left.data[left_offset], right.data[right_offset])
        })
        .collect();

    Array {
        data,
        shape,
        strides,
    }
}

pub(crate) fn ternary_map<F>(first: &Array, second: &Array, third: &Array, mut op: F) -> Array
where
    F: FnMut(f64, f64, f64) -> f64,
{
    let shape = broadcast_shapes(&[&first.shape, &second.shape, &third.shape]);
    let strides = compute_strides(&shape);
    let first_strides = broadcast_strides(&first.shape, &first.strides, &shape);
    let second_strides = broadcast_strides(&second.shape, &second.strides, &shape);
    let third_strides = broadcast_strides(&third.shape, &third.strides, &shape);
    let size = compute_size(&shape);

    let data = (0..size)
        .map(|flat_index| {
            let first_offset = offsets_for_broadcast(&shape, &strides, &first_strides, flat_index);
            let second_offset =
                offsets_for_broadcast(&shape, &strides, &second_strides, flat_index);
            let third_offset = offsets_for_broadcast(&shape, &strides, &third_strides, flat_index);
            op(
                first.data[first_offset],
                second.data[second_offset],
                third.data[third_offset],
            )
        })
        .collect();

    Array {
        data,
        shape,
        strides,
    }
}

pub(crate) fn reduce_axis<F>(array: &Array, axis: usize, init: f64, mut op: F) -> Array
where
    F: FnMut(f64, f64) -> f64,
{
    normalize_axis(axis, array.ndim());

    let shape = reduced_shape(&array.shape, axis);
    let strides = compute_strides(&shape);
    let mut data = vec![init; compute_size(&shape)];

    for (flat_index, &value) in array.data.iter().enumerate() {
        let mut remainder = flat_index;
        let mut reduced_offset = 0;
        let mut reduced_axis = 0;

        for current_axis in 0..array.ndim() {
            let coordinate = remainder / array.strides[current_axis];
            remainder %= array.strides[current_axis];

            if current_axis != axis {
                reduced_offset += coordinate * strides[reduced_axis];
                reduced_axis += 1;
            }
        }

        data[reduced_offset] = op(data[reduced_offset], value);
    }

    Array {
        data,
        shape,
        strides,
    }
}
