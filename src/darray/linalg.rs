use super::Array;
use super::DArrayError;
use super::utils::{compute_strides, dot_simd};
use matrixmultiply::dgemm;
use rayon::prelude::*;

impl Array {
    /// Computes the NumPy-style dot product for common 1-D and 2-D cases.
    pub fn dot(&self, other: &Self) -> Result<Self, DArrayError> {
        match (self.ndim(), other.ndim()) {
            (1, 1) => {
                if self.shape[0] != other.shape[0] {
                    return Err(DArrayError::DotShapeMismatch {
                        left: self.shape.to_vec(),
                        right: other.shape.to_vec(),
                    });
                }
                let value = dot_simd(&self.data, &other.data);
                Ok(Self::scalar(value))
            }
            (2, 1) => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                if cols != other.shape[0] {
                    return Err(DArrayError::DotShapeMismatch {
                        left: self.shape.to_vec(),
                        right: other.shape.to_vec(),
                    });
                }

                let data = if rows * cols >= 16_384 {
                    (0..rows)
                        .into_par_iter()
                        .map(|row| {
                            let start = row * cols;
                            let end = start + cols;
                            dot_simd(&self.data[start..end], &other.data)
                        })
                        .collect::<Vec<_>>()
                } else {
                    (0..rows)
                        .map(|row| {
                            let start = row * cols;
                            let end = start + cols;
                            dot_simd(&self.data[start..end], &other.data)
                        })
                        .collect::<Vec<_>>()
                };

                Ok(Self::from_shape_vec(&[rows], data))
            }
            (2, 2) => self.matmul(other),
            _ => Err(DArrayError::DotUnsupportedDimensions {
                left: self.shape.to_vec(),
                right: other.shape.to_vec(),
            }),
        }
    }

    /// Computes matrix multiplication for two 2-D arrays.
    pub fn matmul(&self, other: &Self) -> Result<Self, DArrayError> {
        if self.ndim() != 2 {
            return Err(DArrayError::MatmulInvalidLeftShape(self.shape.to_vec()));
        }
        if other.ndim() != 2 {
            return Err(DArrayError::MatmulInvalidRightShape(other.shape.to_vec()));
        }

        let rows = self.shape[0];
        let shared = self.shape[1];
        let other_shared = other.shape[0];
        let cols = other.shape[1];

        if shared != other_shared {
            return Err(DArrayError::MatmulShapeMismatch {
                left: self.shape.to_vec(),
                right: other.shape.to_vec(),
            });
        }

        let mut data = vec![0.0; rows * cols];
        unsafe {
            dgemm(
                rows,
                shared,
                cols,
                1.0,
                self.data.as_ptr(),
                shared as isize,
                1,
                other.data.as_ptr(),
                cols as isize,
                1,
                0.0,
                data.as_mut_ptr(),
                cols as isize,
                1,
            );
        }

        Ok(Self {
            data,
            shape: [rows, cols].into_iter().collect(),
            strides: compute_strides(&[rows, cols]),
        })
    }

    /// Computes the vector outer product.
    pub fn outer(&self, other: &Self) -> Result<Self, DArrayError> {
        if self.ndim() != 1 {
            return Err(DArrayError::OuterInvalidLeftShape(self.shape.to_vec()));
        }
        if other.ndim() != 1 {
            return Err(DArrayError::OuterInvalidRightShape(other.shape.to_vec()));
        }

        let rows = self.shape[0];
        let cols = other.shape[0];
        let data = self
            .data
            .iter()
            .flat_map(|left| other.data.iter().map(move |right| left * right))
            .collect::<Vec<_>>();

        Ok(Self::from_shape_vec(&[rows, cols], data))
    }

    /// Computes the dot product of flattened arrays.
    pub fn vdot(&self, other: &Self) -> Result<Self, DArrayError> {
        if self.len() != other.len() {
            return Err(DArrayError::VdotShapeMismatch {
                left_len: self.len(),
                right_len: other.len(),
            });
        }
        let value = dot_simd(&self.data, &other.data);
        Ok(Self::scalar(value))
    }

    /// Creates or extracts a diagonal.
    pub fn diag(&self, diagonal: isize) -> Result<Self, DArrayError> {
        let row_offset = if diagonal < 0 {
            (-diagonal) as usize
        } else {
            0
        };
        let col_offset = if diagonal > 0 { diagonal as usize } else { 0 };

        match self.ndim() {
            1 => {
                let size = self.shape[0];
                let rows = size + row_offset;
                let cols = size + col_offset;
                let mut array = Self::zeros(&[rows, cols]);

                for (index, &value) in self.data.iter().enumerate() {
                    let row = index + row_offset;
                    let col = index + col_offset;
                    array.set(&[row, col], value);
                }

                Ok(array)
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                let start_row = row_offset;
                let start_col = col_offset;

                if start_row >= rows || start_col >= cols {
                    return Ok(Self::array(&[]));
                }

                let len = (rows - start_row).min(cols - start_col);
                let data = (0..len)
                    .map(|index| self.get(&[start_row + index, start_col + index]))
                    .collect::<Vec<_>>();
                Ok(Self::array(&data))
            }
            _ => Err(DArrayError::DiagInvalidShape(self.shape.to_vec())),
        }
    }

    /// Returns a diagonal matrix built from the flattened array.
    pub fn diagflat(&self, diagonal: isize) -> Result<Self, DArrayError> {
        self.flatten().diag(diagonal)
    }

    /// Sums over a matrix diagonal.
    pub fn trace(&self, offset: isize) -> Result<f64, DArrayError> {
        if self.ndim() != 2 {
            return Err(DArrayError::TraceInvalidShape(self.shape.to_vec()));
        }
        Ok(self.diag(offset)?.sum())
    }
}
