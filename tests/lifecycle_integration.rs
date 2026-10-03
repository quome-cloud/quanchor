//! Paper 012 — end-to-end lifecycle deployment-gate integration. Builds a signed
//! passport, writes it to disk, and drives `gate::decide` over (a) a clean
//! in-envelope live config and (b) an out-of-envelope one, asserting the gate
//! admits / blocks and that the decision is stamped into the 003 audit chain.

use qfire::audit::AuditSink;
use qfire::lifecycle::fingerprint::{Component, Fingerprint};
use qfire::lifecycle::gate::{decide, stamp, Enforce};
use qfire::lifecycle::passport::{EnvelopeSpec, Passport, SignedPassport};
use qfire::lifecycle::pccp::{ChangePolicy, ComponentRule, Pccp};
use qfire::lifecycle::{AgentMetadata, AutonomyLevel, ClinicalTask, OutputType, Severity};
use ed25519_dalek::SigningKey;

fn build_passport() -> Passport {
    Passport {
        agent_id: "sepsis-triage".into(),
        version: "2.1.0".into(),
        ruleset_version: "2026.06".into(),
        metadata: AgentMetadata {
            intended_use: "ED sepsis triage prioritisation".into(),
            clinical_task: ClinicalTask::Drive,
            output_type: OutputType::Recommendation,
            autonomy: AutonomyLevel::Advisory,
            condition_severity: Severity::Serious,
            patient_facing: false,
            tools: vec!["vitals_lookup".into(), "ews_score".into()],
        },
        classifications: vec![],
        // PCCP: prompt may change freely; weights bounded to 5%; nothing else.
        pccp: Pccp {
            allowed: vec![
                ComponentRule { component: Component::Prompt, change: ChangePolicy::AnyChange },
                ComponentRule {
                    component: Component::Weights,
                    change: ChangePolicy::BoundedMagnitude { max_fraction: 0.05 },
                },
            ],
        },
        deployed_fingerprint: Fingerprint::of(b"w0", b"p0", b"t0", b"d0"),
        not_after: 4_000_000_000,
        autonomy_envelope: EnvelopeSpec {
            max_autonomous_risk_tier: 2,
            max_autonomous_fraction: 0.25,
            window: 200,
        },
    }
}

fn write_signed(p: &Passport, dir: &std::path::Path) -> std::path::PathBuf {
    let signed = p.sign(&SigningKey::from_bytes(&[3u8; 32])).unwrap();
    let path = dir.join("passport.json");
    std::fs::write(&path, serde_json::to_string(&signed).unwrap()).unwrap();
    path
}

#[test]
fn clean_in_envelope_config_admits() {
    let dir = tempfile::tempdir().unwrap();
    let p = build_passport();
    let path = write_signed(&p, dir.path());

    let signed: SignedPassport =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    // live config: prompt changed only (in-envelope: AnyChange).
    let live = Fingerprint::of(b"w0", b"p-NEW", b"t0", b"d0");
    let report = decide(&signed, &live, &["sepsis-triage".into()], 1_000_000, Enforce::Block);

    assert!(report.admitted, "clean in-envelope config must admit: {:?}", report.reasons);
    assert!(report.reasons.is_empty());
    assert_eq!(report.passport_version, "2.1.0");
}

#[test]
fn out_of_envelope_config_blocks_and_stamps() {
    use qfire::audit::chain::{parse_line, EntryKind};
    use qfire::audit::store::{Mode, StoreCfg, TamperEvidentLog};

    let dir = tempfile::tempdir().unwrap();
    let p = build_passport();
    let path = write_signed(&p, dir.path());
    let signed: SignedPassport =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    // live config: tools changed — tools is unlisted in the PCCP ⇒ out of envelope.
    let live = Fingerprint::of(b"w0", b"p0", b"t-NEW-TOOL", b"d0");
    let report = decide(&signed, &live, &["sepsis-triage".into()], 1_000_000, Enforce::Block);
    assert!(!report.admitted);
    assert!(!report.envelope.in_envelope);
    assert!(report.reasons.iter().any(|r| r.contains("PCCP envelope")));

    // The blocked decision is still stamped into the audit chain.
    let log = TamperEvidentLog::open(StoreCfg {
        path: dir.path().join("audit.jsonl"),
        mode: Mode::Chained,
        signer: None,
        anchor: None,
        batch: 4,
        fail_open: false,
    })
    .unwrap();
    let sink = AuditSink::Chained(log);
    stamp(&report, &sink).unwrap();
    drop(sink);

    let text = std::fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
    let e = parse_line(text.lines().nth(1).unwrap()).unwrap();
    assert_eq!(e.kind, EntryKind::Admission);
    assert_eq!(e.body["event"], "lifecycle_admission");
    assert_eq!(e.body["passport_version"], "2.1.0");
    assert_eq!(e.body["admitted"], false);
}
