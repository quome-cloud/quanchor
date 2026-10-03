//! Aggregate-corpus loading, distinct-vuln dedup, and coverage scoring.
//!
//! The HAARF-RedTeam corpus is a JSONL file of [`AttackRecord`]s. Coverage is
//! reported with an explicit built/unbuilt split so the 012–014 gap is visible,
//! never silently dropped (the proposal's no-silent-caps ethos).

use crate::redteam::{AttackRecord, TargetLayer, Vulnerability};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::Path;

/// Load attacks from a JSONL file (one [`AttackRecord`] per line). Blank lines skipped.
pub fn load_corpus(path: impl AsRef<Path>) -> std::io::Result<Vec<AttackRecord>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: AttackRecord = serde_json::from_str(line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        out.push(rec);
    }
    Ok(out)
}

/// Distinct vulnerabilities present in a set of attacks (dedup by vuln-key).
pub fn distinct_vulns(attacks: &[AttackRecord]) -> BTreeSet<Vulnerability> {
    attacks.iter().map(|a| a.vuln()).collect()
}

/// Coverage of the 13 layers, split built/unbuilt, plus distinct ATLAS techniques.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Coverage {
    pub layers_total: usize,
    pub layers_built: usize,
    pub layers_with_attacks: usize,
    pub built_with_attacks: usize,
    pub unbuilt_with_attacks: usize,
    pub atlas_techniques: usize,
    /// Names of built layers that have NO attack (under-tested built layers).
    pub built_untested: Vec<String>,
    /// Names of unbuilt layers represented in the corpus (gap-tagged).
    pub unbuilt_tested: Vec<String>,
}

/// Score corpus coverage over all 13 [`TargetLayer`]s.
pub fn coverage(attacks: &[AttackRecord]) -> Coverage {
    let layers_with: BTreeSet<TargetLayer> = attacks.iter().map(|a| a.layer).collect();
    let atlas: BTreeSet<&str> = attacks.iter().map(|a| a.atlas.as_str()).collect();

    let built_with_attacks = layers_with.iter().filter(|l| l.built()).count();
    let unbuilt_with_attacks = layers_with.iter().filter(|l| !l.built()).count();

    let built_untested = TargetLayer::ALL
        .iter()
        .filter(|l| l.built() && !layers_with.contains(l))
        .map(|l| l.to_string())
        .collect();
    let unbuilt_tested = TargetLayer::ALL
        .iter()
        .filter(|l| !l.built() && layers_with.contains(l))
        .map(|l| l.to_string())
        .collect();

    Coverage {
        layers_total: TargetLayer::ALL.len(),
        layers_built: TargetLayer::ALL.iter().filter(|l| l.built()).count(),
        layers_with_attacks: layers_with.len(),
        built_with_attacks,
        unbuilt_with_attacks,
        atlas_techniques: atlas.len(),
        built_untested,
        unbuilt_tested,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redteam::Control;

    fn atk(layer: TargetLayer, atlas: &str) -> AttackRecord {
        AttackRecord {
            id: format!("{layer}-{atlas}"),
            control: Control::C3,
            atlas: atlas.into(),
            layer,
            built: layer.built(),
            payload: "p".into(),
            expected_block: true,
            seed_round: 0,
            lineage: None,
        }
    }

    #[test]
    fn dedup_collapses_same_vuln_key() {
        let v = vec![
            atk(TargetLayer::Firewall, "AML.T0051"),
            atk(TargetLayer::Firewall, "AML.T0051"),
        ];
        assert_eq!(distinct_vulns(&v).len(), 1);
    }

    #[test]
    fn coverage_splits_built_and_unbuilt() {
        let v = vec![
            atk(TargetLayer::Firewall, "AML.T0051"),
            atk(TargetLayer::Consent, "AML.T0054"),
        ];
        let c = coverage(&v);
        assert_eq!(c.layers_total, 13);
        assert_eq!(c.layers_built, 10);
        assert_eq!(c.built_with_attacks, 1);
        assert_eq!(c.unbuilt_with_attacks, 1);
        assert_eq!(c.atlas_techniques, 2);
        assert!(c.unbuilt_tested.contains(&"Consent".to_string()));
    }

    #[test]
    fn load_corpus_round_trips_jsonl() {
        let dir = std::env::temp_dir().join("redteam_corpus_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.jsonl");
        let recs = vec![atk(TargetLayer::Firewall, "AML.T0051"), atk(TargetLayer::Policy, "AML.T0052")];
        let body: String = recs
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, format!("{body}\n\n")).unwrap();
        let loaded = load_corpus(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(distinct_vulns(&loaded).len(), 2);
    }
}
