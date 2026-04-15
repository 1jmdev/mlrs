use smallvec::SmallVec;

/// A dense, row-major, multi-dimensional array of `f64` values.
///
/// The array stores flattened data alongside a shape and precomputed strides.
/// Most methods return new arrays, which keeps usage straightforward and close
/// to the immutable style commonly used with NumPy expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub(crate) data: Vec<f64>,
    pub(crate) shape: SmallVec<[usize; 6]>,
    pub(crate) strides: SmallVec<[usize; 6]>,
}

impl Array {
    /// Returns a deep copy of the array.
    pub fn copy(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Returns the flattened storage backing the array.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns mutable access to the flattened storage backing the array.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns the row-major shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the row-major strides of the array.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the total number of stored elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the flattened storage has zero elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns whether the array is a scalar.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Returns whether the array is one-dimensional.
    pub fn is_vector(&self) -> bool {
        self.shape.len() == 1
    }

    /// Returns whether the array is two-dimensional.
    pub fn is_matrix(&self) -> bool {
        self.shape.len() == 2
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns a copied `Vec<f64>` with the flattened contents.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }
}
