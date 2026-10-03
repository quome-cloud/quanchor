//! Paper 015 — off-hot-path continuous red-teaming, IR & resilience meta-layer.
//!
//! The redteam layer never touches the request path. It replays a synthetic
//! attack corpus against a modeled gateway defense (discovery + regression),
//! consumes the 003 decision stream to drive incident runbooks, and injects
//! faults to verify resilience invariants. All headline experiments are
//! deterministic and model-free; an optional `--live` slice confirms via Ollama.

pub mod chaos;
pub mod corpus;
pub mod incident;
pub mod regression;
pub mod runner;

use serde::{Deserialize, Serialize};
use std::fmt;

/// HAARF control family an attack targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Control {
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
}

/// A qfire layer under test. The last three are NOT yet built on master and are
/// always-bypass coverage-gap tags (reported, never silently dropped).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetLayer {
    Firewall,
    Audit,
    Admission,
    Policy,
    Egress,
    Retrieval,
    Identity,
    Oversight,
    Equity,
    Monitor,
    // ── NOT built (012–014): always-bypass coverage-gap tags ──
    Lifecycle,
    Consent,
    Device,
}

impl TargetLayer {
    /// Whether the layer is implemented on master (001–011 built; 012–014 not).
    pub fn built(self) -> bool {
        !matches!(
            self,
            TargetLayer::Lifecycle | TargetLayer::Consent | TargetLayer::Device
        )
    }

    /// All 13 layers under (potential) test.
    pub const ALL: [TargetLayer; 13] = [
        TargetLayer::Firewall,
        TargetLayer::Audit,
        TargetLayer::Admission,
        TargetLayer::Policy,
        TargetLayer::Egress,
        TargetLayer::Retrieval,
        TargetLayer::Identity,
        TargetLayer::Oversight,
        TargetLayer::Equity,
        TargetLayer::Monitor,
        TargetLayer::Lifecycle,
        TargetLayer::Consent,
        TargetLayer::Device,
    ];
}

/// One synthetic attack in the HAARF-RedTeam corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackRecord {
    pub id: String,
    pub control: Control,
    /// MITRE ATLAS technique id, e.g. "AML.T0051".
    pub atlas: String,
    pub layer: TargetLayer,
    /// Redundant with `layer.built()`; carried for the dataset reader.
    #[serde(default)]
    pub built: bool,
    pub payload: String,
    pub expected_block: bool,
    /// 0 = surfaced by a one-shot run; >0 = continuous-only (mutation round).
    #[serde(default)]
    pub seed_round: u32,
    /// Parent attack id (mutation provenance).
    #[serde(default)]
    pub lineage: Option<String>,
}

/// The distinct-vulnerability key. Two bypasses with the same key are the same vuln.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Vulnerability {
    pub control: Control,
    pub atlas: String,
    pub layer: TargetLayer,
}

impl AttackRecord {
    pub fn vuln(&self) -> Vulnerability {
        Vulnerability {
            control: self.control,
            atlas: self.atlas.clone(),
            layer: self.layer,
        }
    }
}

/// Outcome of probing one attack against the modeled gateway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RedTeamVerdict {
    pub attack_id: String,
    pub blocked: bool,
    /// A vulnerability iff the gateway failed to block an attack it should have.
    pub bypass: bool,
}

/// One event in an incident stream (a 003-style decision trace with injected attacks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentEvent {
    /// Monotonic stream time-step.
    pub step: u64,
    pub agent_id: String,
    /// Present iff this step is part of a live attack.
    #[serde(default)]
    pub attack_signature: Option<String>,
    /// Detector signal in [0,1].
    #[serde(default)]
    pub score: f64,
}

/// A containment/recovery action a runbook can take (modeled effect on the stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunbookAction {
    IsolateAgent,
    RevokeCapability,
    TripBreaker,
    Snapshot,
}

/// A socio-technical fault injected at the gateway boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultKind {
    Latency,
    DependencyLoss,
    Partition,
}

/// A resilience invariant checked under fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResilienceInvariant {
    /// I1: never allow under fault what would otherwise be blocked.
    NoUnsafeAction,
    /// I2: fail closed, shed load — availability within the declared envelope.
    BoundedDegradation,
}

impl fmt::Display for TargetLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A deterministic, model-free probe of one attack against the gateway defense.
pub trait AttackProbe {
    fn probe(&self, a: &AttackRecord) -> RedTeamVerdict;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbuilt_layers_report_not_built() {
        assert!(!TargetLayer::Consent.built());
        assert!(!TargetLayer::Lifecycle.built());
        assert!(!TargetLayer::Device.built());
        assert!(TargetLayer::Policy.built());
        assert_eq!(TargetLayer::ALL.iter().filter(|l| l.built()).count(), 10);
        assert_eq!(TargetLayer::ALL.len(), 13);
    }

    #[test]
    fn attack_record_serde_round_trip() {
        let a = AttackRecord {
            id: "a1".into(),
            control: Control::C3,
            atlas: "AML.T0051".into(),
            layer: TargetLayer::Firewall,
            built: true,
            payload: "ignore prior".into(),
            expected_block: true,
            seed_round: 0,
            lineage: None,
        };
        let s = serde_json::to_string(&a).unwrap();
        let b: AttackRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(a.vuln(), b.vuln());
        assert_eq!(b.layer, TargetLayer::Firewall);
    }

    #[test]
    fn vuln_key_ignores_payload_and_id() {
        let base = |id: &str, p: &str| AttackRecord {
            id: id.into(),
            control: Control::C3,
            atlas: "AML.T0051".into(),
            layer: TargetLayer::Firewall,
            built: true,
            payload: p.into(),
            expected_block: true,
            seed_round: 0,
            lineage: None,
        };
        assert_eq!(base("a", "x").vuln(), base("b", "y").vuln());
    }
}
