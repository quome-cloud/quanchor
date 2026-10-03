//! The admission-time deployment gate. At startup / agent-registration time the
//! gateway runs `decide` beside the 004 AIBOM gate: an agent may register only if
//! its passport is validly signed and unexpired, its `agent_id` is in the 008
//! registry, and its **live** configuration fingerprint is in-envelope versus the
//! passport's deployed fingerprint per the PCCP. Fail-closed under `Enforce::Block`.
//! `stamp` writes the passport version + verdict into the 003 audit chain (C2.6).

use super::fingerprint::{self, Fingerprint};
use super::pccp::{self, EnvelopeVerdict};
use super::passport::SignedPassport;
use crate::audit::AuditSink;
use crate::Result;
use serde::{Deserialize, Serialize};

/// How strict the lifecycle gate is (mirrors `admission::gate::Enforce`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Enforce {
    /// Any violation blocks registration (enclave default).
    Block,
    /// Log violations but proceed (dev default).
    Warn,
    /// Skip the lifecycle gate entirely.
    Off,
}

impl Default for Enforce {
    fn default() -> Self {
        Enforce::Warn
    }
}

/// The outcome of a lifecycle admission check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleReport {
    pub admitted: bool,
    pub agent_id: String,
    pub passport_version: String,
    /// Why the agent failed the gate (empty iff fully clean).
    pub reasons: Vec<String>,
    pub envelope: EnvelopeVerdict,
    pub enforce: Enforce,
}

/// Decide lifecycle admission (pure; no I/O). `registered_ids` is the set of
/// agent_ids the 008 registry has verified; `now` is unix seconds.
pub fn decide(
    signed: &SignedPassport,
    live: &Fingerprint,
    registered_ids: &[String],
    now: i64,
    enforce: Enforce,
) -> LifecycleReport {
    if enforce == Enforce::Off {
        return LifecycleReport {
            admitted: true,
            agent_id: String::new(),
            passport_version: String::new(),
            reasons: vec![],
            envelope: EnvelopeVerdict { in_envelope: true, violations: vec![] },
            enforce,
        };
    }

    let mut reasons = Vec::new();

    // 1. Signature + expiry (fail-closed).
    let parsed = signed.verify_at(now);
    let (agent_id, version, envelope) = match parsed {
        Err(e) => {
            reasons.push(format!("passport invalid: {e}"));
            // No trusted body to read; block (or warn) with an empty envelope.
            return finalize(reasons, String::new(), String::new(),
                EnvelopeVerdict { in_envelope: false, violations: vec![] }, enforce);
        }
        Ok(p) => {
            // 2. Registry membership (links 008).
            if !registered_ids.iter().any(|id| id == &p.agent_id) {
                reasons.push(format!("agent_id {} not in signed registry", p.agent_id));
            }
            // 3. Fingerprint diff → PCCP envelope (the L3 decision).
            let deltas = fingerprint::diff(&p.deployed_fingerprint, live);
            let verdict = pccp::evaluate(&deltas, &p.pccp);
            if !verdict.in_envelope {
                reasons.push(format!(
                    "live config diverges beyond PCCP envelope: {:?}",
                    verdict.violations
                ));
            }
            (p.agent_id, p.version, verdict)
        }
    };

    finalize(reasons, agent_id, version, envelope, enforce)
}

fn finalize(
    reasons: Vec<String>,
    agent_id: String,
    passport_version: String,
    envelope: EnvelopeVerdict,
    enforce: Enforce,
) -> LifecycleReport {
    let clean = reasons.is_empty();
    let admitted = match enforce {
        Enforce::Block => clean,
        Enforce::Warn | Enforce::Off => true,
    };
    LifecycleReport { admitted, agent_id, passport_version, reasons, envelope, enforce }
}

/// Stamp the lifecycle decision into the 003 audit chain (no-op on a plain sink).
pub fn stamp(report: &LifecycleReport, audit: &AuditSink) -> Result<()> {
    let body = serde_json::json!({
        "event": "lifecycle_admission",
        "agent_id": report.agent_id,
        "passport_version": report.passport_version,
        "admitted": report.admitted,
        "in_envelope": report.envelope.in_envelope,
        "reasons": report.reasons.len(),
    });
    audit.append_admission_json(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::fingerprint::Fingerprint;
    use crate::lifecycle::passport::{EnvelopeSpec, Passport};
    use crate::lifecycle::pccp::{ChangePolicy, ComponentRule, Pccp};
    use crate::lifecycle::fingerprint::Component;
    use crate::lifecycle::{AgentMetadata, AutonomyLevel, ClinicalTask, OutputType, Severity};
    use ed25519_dalek::SigningKey;

    fn base_passport() -> Passport {
        Passport {
            agent_id: "agent-1".into(),
            version: "1.0.0".into(),
            ruleset_version: "2026.06".into(),
            metadata: AgentMetadata {
                intended_use: "triage".into(),
                clinical_task: ClinicalTask::Drive,
                output_type: OutputType::Recommendation,
                autonomy: AutonomyLevel::Advisory,
                condition_severity: Severity::Serious,
                patient_facing: false,
                tools: vec!["lookup".into()],
            },
            classifications: vec![],
            // Prompt may change freely; everything else forbidden.
            pccp: Pccp {
                allowed: vec![ComponentRule {
                    component: Component::Prompt,
                    change: ChangePolicy::AnyChange,
                }],
            },
            deployed_fingerprint: Fingerprint::of(b"w", b"p", b"t", b"d"),
            not_after: 10_000,
            autonomy_envelope: EnvelopeSpec {
                max_autonomous_risk_tier: 2,
                max_autonomous_fraction: 0.3,
                window: 100,
            },
        }
    }

    fn signed(p: &Passport) -> SignedPassport {
        p.sign(&SigningKey::from_bytes(&[7u8; 32])).unwrap()
    }

    #[test]
    fn clean_passport_admitted() {
        let p = base_passport();
        let s = signed(&p);
        // live differs only in prompt (in-envelope).
        let live = Fingerprint::of(b"w", b"p2", b"t", b"d");
        let r = decide(&s, &live, &["agent-1".into()], 9_000, Enforce::Block);
        assert!(r.admitted);
        assert!(r.reasons.is_empty());
        assert_eq!(r.passport_version, "1.0.0");
    }

    #[test]
    fn unregistered_agent_blocked() {
        let p = base_passport();
        let s = signed(&p);
        let live = p.deployed_fingerprint.clone();
        let r = decide(&s, &live, &[], 9_000, Enforce::Block);
        assert!(!r.admitted);
        assert!(r.reasons.iter().any(|x| x.contains("not in signed registry")));
    }

    #[test]
    fn expired_passport_blocked() {
        let p = base_passport();
        let s = signed(&p);
        let live = p.deployed_fingerprint.clone();
        let r = decide(&s, &live, &["agent-1".into()], 20_000, Enforce::Block);
        assert!(!r.admitted);
        assert!(r.reasons.iter().any(|x| x.contains("expired")));
    }

    #[test]
    fn bad_signature_blocked() {
        let p = base_passport();
        let mut s = signed(&p);
        s.passport_json = s.passport_json.replace("triage", "tamper");
        let live = p.deployed_fingerprint.clone();
        let r = decide(&s, &live, &["agent-1".into()], 9_000, Enforce::Block);
        assert!(!r.admitted);
        assert!(r.reasons.iter().any(|x| x.contains("invalid")));
    }

    #[test]
    fn out_of_envelope_blocked() {
        let p = base_passport();
        let s = signed(&p);
        // weights changed — Forbidden by default (not listed) ⇒ out of envelope.
        let live = Fingerprint::of(b"W2", b"p", b"t", b"d");
        let r = decide(&s, &live, &["agent-1".into()], 9_000, Enforce::Block);
        assert!(!r.admitted);
        assert!(!r.envelope.in_envelope);
        assert!(r.reasons.iter().any(|x| x.contains("PCCP envelope")));
    }

    #[test]
    fn warn_admits_with_reasons() {
        let p = base_passport();
        let s = signed(&p);
        let live = Fingerprint::of(b"W2", b"p", b"t", b"d");
        let r = decide(&s, &live, &[], 9_000, Enforce::Warn);
        assert!(r.admitted); // warn admits
        assert!(!r.reasons.is_empty()); // but records the violations
    }

    #[test]
    fn off_admits_unconditionally() {
        let p = base_passport();
        let s = signed(&p);
        let live = Fingerprint::of(b"W2", b"p2", b"t2", b"d2");
        let r = decide(&s, &live, &[], 99_999, Enforce::Off);
        assert!(r.admitted);
    }

    #[test]
    fn stamp_writes_passport_version() {
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
        let p = base_passport();
        let s = signed(&p);
        let live = p.deployed_fingerprint.clone();
        let r = decide(&s, &live, &["agent-1".into()], 9_000, Enforce::Block);
        stamp(&r, &sink).unwrap();
        drop(sink);
        let text = std::fs::read_to_string(dir.path().join("a.jsonl")).unwrap();
        let e = parse_line(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(e.kind, EntryKind::Admission);
        assert_eq!(e.body["event"], "lifecycle_admission");
        assert_eq!(e.body["passport_version"], "1.0.0");
    }
}
