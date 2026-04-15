use super::Array;
use super::utils::{compute_size, compute_strides, validate_shape_data};
use smallvec::SmallVec;

impl Array {
    /// Creates an array from flattened data and an explicit shape.
    ///
    /// # Examples
    /// ```
    /// use mlrs::darray::Array;
    ///
    /// let array = Array::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(array.shape(), &[2, 2]);
    /// ```
    pub fn from_shape_vec(shape: &[usize], data: Vec<f64>) -> Self {
        validate_shape_data(shape, data.len());

        Self {
            data,
            shape: SmallVec::from_slice(shape),
            strides: compute_strides(shape),
        }
    }

    /// Creates a one-dimensional array from a slice.
    pub fn array(data: &[f64]) -> Self {
        Self::from_shape_vec(&[data.len()], data.to_vec())
    }

    /// Creates a scalar array.
    pub fn scalar(value: f64) -> Self {
        Self::from_shape_vec(&[], vec![value])
    }

    /// Creates a new array filled with a specific value.
    pub fn full(shape: &[usize], fill_value: f64) -> Self {
        Self::from_shape_vec(shape, vec![fill_value; compute_size(shape)])
    }

    /// Creates a new array filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(shape, 0.0)
    }

    /// Creates a new array filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, 1.0)
    }

    /// Creates a new array intended for later assignment.
    ///
    /// NumPy leaves `empty` uninitialized. This implementation fills with `NaN`
    /// instead, which keeps the API safe while still making accidental reads
    /// easy to spot.
    pub fn empty(shape: &[usize]) -> Self {
        Self::full(shape, f64::NAN)
    }

    /// Creates a one-dimensional array with values in `[start, stop)`.
    pub fn arange(start: f64, stop: f64, step: f64) -> Self {
        assert!(step != 0.0, "arange step must not be zero");

        let estimated_len = if (step > 0.0 && start < stop) || (step < 0.0 && start > stop) {
            (((stop - start) / step).abs().ceil() as usize).saturating_add(1)
        } else {
            0
        };
        let mut data = Vec::with_capacity(estimated_len);
        let mut current = start;

        if step > 0.0 {
            while current < stop {
                data.push(current);
                current += step;
            }
        } else {
            while current > stop {
                data.push(current);
                current += step;
            }
        }

        Self::array(&data)
    }

    /// Creates evenly spaced numbers across an interval.
    pub fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Self {
        if num == 0 {
            return Self::array(&[]);
        }
        if num == 1 {
            return Self::array(&[start]);
        }

        let divisor = if endpoint { num - 1 } else { num } as f64;
        let step = (stop - start) / divisor;
        let mut data = (0..num)
            .map(|index| start + step * index as f64)
            .collect::<Vec<_>>();

        if endpoint {
            data[num - 1] = stop;
        }

        Self::array(&data)
    }

    /// Creates a 2-D array with ones on a diagonal and zeros elsewhere.
    pub fn eye(rows: usize, cols: usize, diagonal: isize) -> Self {
        let mut array = Self::zeros(&[rows, cols]);

        for row in 0..rows {
            let column = row as isize + diagonal;
            if (0..cols as isize).contains(&column) {
                let index = row * cols + column as usize;
                array.data[index] = 1.0;
            }
        }

        array
    }

    /// Creates a square identity matrix.
    pub fn identity(size: usize) -> Self {
        Self::eye(size, size, 0)
    }

    /// Creates a zero-filled array matching another array's shape.
    pub fn zeros_like(other: &Self) -> Self {
        Self::zeros(&other.shape)
    }

    /// Creates a one-filled array matching another array's shape.
    pub fn ones_like(other: &Self) -> Self {
        Self::ones(&other.shape)
    }

    /// Creates a filled array matching another array's shape.
    pub fn full_like(other: &Self, fill_value: f64) -> Self {
        Self::full(&other.shape, fill_value)
    }

    /// Creates a NaN-filled array matching another array's shape.
    pub fn empty_like(other: &Self) -> Self {
        Self::empty(&other.shape)
    }
}
