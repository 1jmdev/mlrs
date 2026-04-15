use wide::f64x4;

pub(crate) const SIMD_LANES: usize = 4;

#[inline]
pub(crate) fn load_f64x4(values: &[f64], offset: usize) -> f64x4 {
    f64x4::from([
        values[offset],
        values[offset + 1],
        values[offset + 2],
        values[offset + 3],
    ])
}

#[inline]
pub(crate) fn store_f64x4(values: &mut [f64], offset: usize, vector: f64x4) {
    let chunk: [f64; SIMD_LANES] = vector.into();
    values[offset..offset + SIMD_LANES].copy_from_slice(&chunk);
}

#[inline]
pub(crate) fn reduce_sum(vector: f64x4) -> f64 {
    let values: [f64; SIMD_LANES] = vector.into();
    values.into_iter().sum()
}

#[inline]
pub(crate) fn reduce_max(vector: f64x4) -> f64 {
    let values: [f64; SIMD_LANES] = vector.into();
    values.into_iter().fold(f64::NEG_INFINITY, f64::max)
}
