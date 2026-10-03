//! k-anonymity / aggregation check (paper 006, X5). A call carrying several quasi-identifiers (DOB,
//! ZIP, sex, ...) can re-identify an individual even when no single field is a direct identifier.
//! Flag a call when the number of quasi-identifier fields present meets/exceeds a threshold.

/// Quasi-identifier field names we treat as re-identifying in combination.
pub const QUASI_IDENTIFIERS: &[&str] = &["dob", "zip", "sex", "age", "race", "ethnicity"];

/// True if `present_fields` contains at least `threshold` known quasi-identifiers.
pub fn kanon_violation(present_fields: &[String], threshold: usize) -> bool {
    let n = present_fields
        .iter()
        .filter(|f| QUASI_IDENTIFIERS.contains(&f.as_str()))
        .count();
    n >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn three_quasi_identifiers_violate_at_threshold_two() {
        let fields = vec!["dob".to_string(), "zip".to_string(), "sex".to_string()];
        assert!(kanon_violation(&fields, 2));
    }
    #[test]
    fn single_quasi_identifier_is_safe_at_threshold_two() {
        assert!(!kanon_violation(&vec!["zip".to_string()], 2));
    }
    #[test]
    fn non_quasi_fields_ignored() {
        let fields = vec!["to".to_string(), "body".to_string()];
        assert!(!kanon_violation(&fields, 2));
    }
}
