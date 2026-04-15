use super::super::Array;

impl Array {
    pub fn nonzero(&self) -> Vec<Self> {
        let coordinates = self.nonzero_coordinates();
        let mut per_axis = (0..self.ndim())
            .map(|_| Vec::with_capacity(coordinates.len()))
            .collect::<Vec<Vec<f64>>>();

        for axis_coordinates in coordinates {
            for (axis, coordinate) in axis_coordinates.into_iter().enumerate() {
                per_axis[axis].push(coordinate);
            }
        }

        per_axis
            .into_iter()
            .map(|axis_values| Self::from_shape_vec(&[axis_values.len()], axis_values))
            .collect()
    }

    pub fn flatnonzero(&self) -> Self {
        let data = self
            .data
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value != 0.0).then_some(index as f64))
            .collect::<Vec<_>>();
        Self::from_shape_vec(&[data.len()], data)
    }

    pub fn argwhere(&self) -> Self {
        let coordinates = self.nonzero_coordinates();
        let count = coordinates.len();
        let data = coordinates.into_iter().flatten().collect::<Vec<_>>();
        Self::from_shape_vec(&[count, self.ndim()], data)
    }

    pub fn unique(&self) -> Self {
        let mut data = self.to_vec();
        data.sort_by(f64::total_cmp);
        data.dedup_by(|left, right| left.total_cmp(right).is_eq());
        Self::from_shape_vec(&[data.len()], data)
    }

    fn nonzero_coordinates(&self) -> Vec<Vec<f64>> {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(flat_index, &value)| (value != 0.0).then(|| self.coordinates(flat_index)))
            .collect()
    }

    fn coordinates(&self, flat_index: usize) -> Vec<f64> {
        let mut remainder = flat_index;
        let mut coordinates = Vec::with_capacity(self.ndim());

        for axis in 0..self.ndim() {
            let coordinate = if self.shape.is_empty() {
                0
            } else {
                remainder / self.strides[axis]
            };
            if !self.shape.is_empty() {
                remainder %= self.strides[axis];
            }
            coordinates.push(coordinate as f64);
        }

        coordinates
    }
}
