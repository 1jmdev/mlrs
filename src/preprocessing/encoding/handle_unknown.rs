/// Selects how `OneHotEncoder` handles unseen categories at transform time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleUnknown {
    /// Rejects unseen categories with an error.
    Error,
    /// Leaves unseen categories as all-zero encoded segments.
    Ignore,
}
