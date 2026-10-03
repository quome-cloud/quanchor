//! The Policy Information Point: fetches clinical context attributes for an
//! authorization request. SyntheticPip loads a JSON attribute store (Synthea-
//! derived: encounters, panels, formulary dose ranges) and caches it. A real
//! EHR/FHIR PIP adapter is the production path (out of scope here).

use crate::Result;
use std::collections::HashMap;
use std::path::Path;

/// Fetches context attributes for (principal, resource).
pub trait PolicyInformationPoint {
    fn attributes(&self, principal: &str, resource: &str) -> serde_json::Value;
}

/// A synthetic attribute store loaded from JSON:
/// { "<principal>|<resource>": { "active_encounter": true, "paneled": true,
///   "formulary": {"morphine": {"max_mg": 30}}, ... }, ... }
pub struct SyntheticPip {
    by_key: HashMap<String, serde_json::Value>,
}

impl SyntheticPip {
    pub fn empty() -> Self {
        SyntheticPip {
            by_key: HashMap::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let map: HashMap<String, serde_json::Value> = serde_json::from_str(&text)?;
        Ok(SyntheticPip { by_key: map })
    }

    fn key(principal: &str, resource: &str) -> String {
        format!("{principal}|{resource}")
    }
}

impl PolicyInformationPoint for SyntheticPip {
    fn attributes(&self, principal: &str, resource: &str) -> serde_json::Value {
        self.by_key
            .get(&Self::key(principal, resource))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn loads_and_returns_attributes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("attrs.json");
        std::fs::write(
            &p,
            r#"{"prescriber|patient-1":{"active_encounter":true,"paneled":true}}"#,
        )
        .unwrap();
        let pip = SyntheticPip::load(&p).unwrap();
        let a = pip.attributes("prescriber", "patient-1");
        assert_eq!(a["active_encounter"], true);
        assert_eq!(a["paneled"], true);
    }

    #[test]
    fn unknown_key_returns_empty_object() {
        let pip = SyntheticPip::empty();
        assert_eq!(pip.attributes("x", "y"), json!({}));
    }
}
