//! The fail-closed admission decision. Given an enumerated AIBOM, the tampered
//! set, and the vulnerability matches, decide whether the gateway may start, and
//! stamp the verified AIBOM digest into the 003 audit chain.

use super::aibom::Aibom;
use super::vuln::VulnMatch;
use crate::audit::AuditSink;
use crate::Result;

/// How strict the gate is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Enforce {
    /// Any tampered or known-vulnerable component blocks startup (enclave default).
    Block,
    /// Log violations but proceed (dev / provenance-gap reality).
    Warn,
    /// Skip admission entirely.
    Off,
}

/// The outcome of an admission check.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AdmissionReport {
    pub admitted: bool,
    pub components: usize,
    pub provenance_gap: f64,
    pub tampered: Vec<String>,
    pub vulnerable: Vec<VulnMatch>,
    pub aibom_digest: String,
    pub enforce: Enforce,
}

/// Decide admission from the gathered evidence (pure; no I/O).
pub fn decide(
    aibom: &Aibom,
    tampered: &[String],
    vulnerable: &[VulnMatch],
    enforce: Enforce,
) -> AdmissionReport {
    let violations = !tampered.is_empty() || !vulnerable.is_empty();
    let admitted = match enforce {
        Enforce::Block => !violations,
        Enforce::Warn | Enforce::Off => true,
    };
    AdmissionReport {
        admitted,
        components: aibom.components.len(),
        provenance_gap: aibom.provenance_gap(),
        tampered: tampered.to_vec(),
        vulnerable: vulnerable.to_vec(),
        aibom_digest: aibom.document_digest(),
        enforce,
    }
}

/// Stamp the admission outcome into the 003 audit chain (no-op on a plain sink).
pub fn stamp(report: &AdmissionReport, audit: &AuditSink) -> Result<()> {
    let body = serde_json::json!({
        "event": "admission",
        "admitted": report.admitted,
        "aibom_digest": report.aibom_digest,
        "components": report.components,
        "provenance_gap": report.provenance_gap,
        "tampered": report.tampered.len(),
        "vulnerable": report.vulnerable.len(),
    });
    audit.append_admission_json(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::aibom::{Aibom, Component, ComponentClass};
    use super::super::vuln::VulnMatch;
    use super::*;

    fn aibom() -> Aibom {
        Aibom {
            components: vec![Component {
                class: ComponentClass::Rule,
                name: "r".into(),
                version: "v".into(),
                digest: Some("d".into()),
                attested: true,
            }],
        }
    }
    fn vuln() -> VulnMatch {
        VulnMatch {
            component: "leftpad".into(),
            version: "1.0".into(),
            advisory_id: "OSV-1".into(),
            severity: "HIGH".into(),
        }
    }

    #[test]
    fn clean_admits_in_all_modes() {
        for e in [Enforce::Block, Enforce::Warn, Enforce::Off] {
            assert!(decide(&aibom(), &[], &[], e).admitted, "{e:?}");
        }
    }

    #[test]
    fn block_refuses_on_tamper_or_vuln_warn_allows() {
        assert!(!decide(&aibom(), &["r".into()], &[], Enforce::Block).admitted);
        assert!(!decide(&aibom(), &[], &[vuln()], Enforce::Block).admitted);
        assert!(decide(&aibom(), &["r".into()], &[vuln()], Enforce::Warn).admitted);
        assert!(decide(&aibom(), &["r".into()], &[vuln()], Enforce::Off).admitted);
    }

    #[test]
    fn stamp_writes_admission_entry() {
        use crate::audit::chain::{parse_line, EntryKind};
        use crate::audit::store::{Mode, StoreCfg, TamperEvidentLog};
        let dir = tempfile::tempdir().unwrap();
        let log = TamperEvidentLog::open(StoreCfg {
            path: dir.path().join("a.jsonl"),
            mode: Mode::Chained,
            signer: None,
            anchor: None,
            batch: 4,
            fail_open: false,
        })
        .unwrap();
        let sink = AuditSink::Chained(log);
        let r = decide(&aibom(), &[], &[], Enforce::Block);
        stamp(&r, &sink).unwrap();
        drop(sink);
        let text = std::fs::read_to_string(dir.path().join("a.jsonl")).unwrap();
        let e = parse_line(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(e.kind, EntryKind::Admission);
        assert_eq!(e.body["event"], "admission");
    }
}
