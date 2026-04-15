/// Selects how `OrdinalEncoder` handles unseen categories at transform time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrdinalHandleUnknown {
    /// Rejects unseen categories with an error.
    Error,
    /// Writes a configured sentinel encoded value for unseen categories.
    UseEncodedValue,
}
