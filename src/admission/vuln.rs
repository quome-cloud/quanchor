//! Vulnerability matching (S2): match AIBOM library components against an
//! advisory feed. The default is a pinned OSV-format snapshot for reproducible
//! benchmarking; a live OSV/GitHub-Advisory adapter is the documented production
//! path.

use super::aibom::Aibom;
use crate::Result;
use std::collections::HashMap;
use std::path::Path;

/// One advisory: an affected package name, the affected version(s), and an id.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Advisory {
    pub id: String,
    pub package: String,
    /// Exact affected versions (snapshot simplification: exact-match set).
    pub versions: Vec<String>,
    pub severity: String,
}

/// A vulnerability finding: the component and the advisory it matched.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VulnMatch {
    pub component: String,
    pub version: String,
    pub advisory_id: String,
    pub severity: String,
}

/// Abstracts a vulnerability feed so the live OSV API can replace the snapshot.
pub trait VulnFeed {
    fn matches(&self, aibom: &Aibom) -> Vec<VulnMatch>;
}

/// A pinned OSV-format snapshot loaded from a JSON file: `[{id, package, versions, severity}]`.
pub struct OsvSnapshot {
    by_package: HashMap<String, Vec<Advisory>>,
}

impl OsvSnapshot {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let advisories: Vec<Advisory> = serde_json::from_str(&text)?;
        Ok(Self::from_advisories(advisories))
    }

    pub fn from_advisories(advisories: Vec<Advisory>) -> Self {
        let mut by_package: HashMap<String, Vec<Advisory>> = HashMap::new();
        for a in advisories {
            by_package.entry(a.package.clone()).or_default().push(a);
        }
        OsvSnapshot { by_package }
    }
}

impl VulnFeed for OsvSnapshot {
    fn matches(&self, aibom: &Aibom) -> Vec<VulnMatch> {
        let mut out = Vec::new();
        for c in &aibom.components {
            if let Some(advs) = self.by_package.get(&c.name) {
                for a in advs {
                    if a.versions.iter().any(|v| v == &c.version) {
                        out.push(VulnMatch {
                            component: c.name.clone(),
                            version: c.version.clone(),
                            advisory_id: a.id.clone(),
                            severity: a.severity.clone(),
                        });
                    }
                }
            }
        }
        out
    }
}

/// Documented production adapter: query osv.dev live. Not implemented (network).
pub struct LiveOsvAdapter;
impl VulnFeed for LiveOsvAdapter {
    fn matches(&self, _aibom: &Aibom) -> Vec<VulnMatch> {
        // Production wires an osv.dev batch query here; offline builds use OsvSnapshot.
        Vec::new()
    }
}

/// Convenience: does any component match a known advisory?
pub fn has_known_vulnerable(aibom: &Aibom, feed: &dyn VulnFeed) -> bool {
    !feed.matches(aibom).is_empty()
}

#[cfg(test)]
mod tests {
    use super::super::aibom::{Component, ComponentClass};
    use super::*;

    fn aibom_with(name: &str, version: &str) -> Aibom {
        Aibom {
            components: vec![Component {
                class: ComponentClass::Library,
                name: name.into(),
                version: version.into(),
                digest: Some("x".into()),
                attested: true,
            }],
        }
    }

    fn snapshot() -> OsvSnapshot {
        OsvSnapshot::from_advisories(vec![Advisory {
            id: "OSV-2026-0001".into(),
            package: "leftpad".into(),
            versions: vec!["1.0.0".into(), "1.0.1".into()],
            severity: "HIGH".into(),
        }])
    }

    #[test]
    fn matches_affected_version() {
        let m = snapshot().matches(&aibom_with("leftpad", "1.0.0"));
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].advisory_id, "OSV-2026-0001");
        assert_eq!(m[0].severity, "HIGH");
    }

    #[test]
    fn ignores_unaffected_version_and_package() {
        assert!(snapshot()
            .matches(&aibom_with("leftpad", "2.0.0"))
            .is_empty());
        assert!(snapshot()
            .matches(&aibom_with("rightpad", "1.0.0"))
            .is_empty());
        assert!(!has_known_vulnerable(
            &aibom_with("leftpad", "9.9"),
            &snapshot()
        ));
        assert!(has_known_vulnerable(
            &aibom_with("leftpad", "1.0.1"),
            &snapshot()
        ));
    }

    #[test]
    fn loads_snapshot_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("osv.json");
        std::fs::write(
            &p,
            r#"[{"id":"OSV-1","package":"foo","versions":["1.2.3"],"severity":"CRITICAL"}]"#,
        )
        .unwrap();
        let snap = OsvSnapshot::load(&p).unwrap();
        assert_eq!(snap.matches(&aibom_with("foo", "1.2.3")).len(), 1);
    }
}
