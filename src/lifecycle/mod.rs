//! Paper 012 — the regulatory passport (lifecycle & change-control). This layer
//! runs at **admission time** (beside the 004 AIBOM gate), not per call: it gates
//! whether an agent may register/deploy on the gateway against a signed, versioned
//! **passport** that declares the agent's cleared risk class and autonomy level
//! (across FDA / EU AI Act / Health Canada / MHRA), carries a machine-readable
//! Predetermined Change Control Plan (PCCP), and pins a fingerprint of every
//! component. A change that diverges from the passport beyond its PCCP envelope is
//! refused; the passport version is stamped into the 003 audit chain (C2.6).
//!
//! Everything here is deterministic (a citation-tagged rule engine + content
//! hashing + ed25519 signing) — no model inference on any path, so every benchmark
//! number reproduces from the corpus alone. Fail-closed.

pub mod classifier;
pub mod fingerprint;
pub mod gate;
pub mod harmonize;
pub mod passport;
pub mod pccp;
pub mod rwe;

use serde::{Deserialize, Serialize};

pub use gate::{Enforce, LifecycleReport};
pub use passport::{EnvelopeSpec, Passport, SignedPassport};

/// Framework-agnostic risk ladder. Each jurisdiction maps its own class names onto
/// this spine (e.g. FDA Class I/II/III ≈ Low/Moderate/High; EU minimal/limited/high).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    Minimal,
    Low,
    Moderate,
    High,
}

/// Autonomy ladder (HAARF C6.1): how much the agent may act without a human in the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutonomyLevel {
    /// Surfaces information; takes no position.
    Informational,
    /// Recommends; a human decides and acts.
    Advisory,
    /// Acts, but every action is reviewed before effect.
    Supervised,
    /// Acts without per-action human review.
    Autonomous,
}

impl AutonomyLevel {
    /// Ordinal tier (0..=3) used for the autonomy-envelope KPI and capping.
    pub fn tier(self) -> u8 {
        match self {
            AutonomyLevel::Informational => 0,
            AutonomyLevel::Advisory => 1,
            AutonomyLevel::Supervised => 2,
            AutonomyLevel::Autonomous => 3,
        }
    }
}

/// The four regulatory frameworks the passport classifies against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Jurisdiction {
    /// US FDA — Software as a Medical Device (SaMD), 510(k)/De Novo, PCCP guidance.
    Fda,
    /// EU AI Act (Annex III high-risk) layered over the MDR.
    EuAiAct,
    /// Health Canada — SaMD guidance (IMDRF-aligned).
    HealthCanada,
    /// UK MHRA — SaMD / "Software and AI as a Medical Device" framework.
    Mhra,
}

impl Jurisdiction {
    pub const ALL: [Jurisdiction; 4] = [
        Jurisdiction::Fda,
        Jurisdiction::EuAiAct,
        Jurisdiction::HealthCanada,
        Jurisdiction::Mhra,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Jurisdiction::Fda => "FDA",
            Jurisdiction::EuAiAct => "EU AI Act",
            Jurisdiction::HealthCanada => "Health Canada",
            Jurisdiction::Mhra => "MHRA",
        }
    }
}

/// What the agent clinically does — the primary driver of risk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalTask {
    /// Surfaces reference information (e.g. drug monograph lookup).
    Inform,
    /// Drives clinical management (triage, prioritisation) without naming a diagnosis.
    Drive,
    /// Diagnoses / detects a condition.
    Diagnose,
    /// Selects or administers treatment.
    Treat,
}

/// The kind of output the agent emits — gates how directly it can cause harm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    Information,
    Recommendation,
    Action,
}

/// Severity of the condition the agent bears on (IMDRF "significance of information").
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    NonSerious,
    Serious,
    Critical,
}

/// Declarative agent metadata — the sole input to classification. Authored at
/// clearance time; the classifier is a pure function of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Free-text intended-use statement (provenance only; not parsed).
    pub intended_use: String,
    pub clinical_task: ClinicalTask,
    pub output_type: OutputType,
    /// Declared autonomy (may be capped upward by the classifier for high-risk action).
    pub autonomy: AutonomyLevel,
    pub condition_severity: Severity,
    pub patient_facing: bool,
    /// Tool inventory (names); part of the change-fingerprint surface.
    pub tools: Vec<String>,
}

/// The outcome of classifying one `AgentMetadata` under one `Jurisdiction`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Classification {
    pub risk_class: RiskClass,
    pub autonomy_level: AutonomyLevel,
    /// Citation-tagged trace of the rules that fired (explainability for reviewers).
    pub rationale: Vec<String>,
}

/// Admission-time configuration (a section of the main Config).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LifecycleCfg {
    pub enforce: Enforce,
    /// Path to the signed passport file ("" disables the lifecycle gate).
    pub passport_path: String,
    /// Pinned classification rule-set version (regulatory rules evolve → pin them).
    pub ruleset_version: String,
}

impl Default for LifecycleCfg {
    fn default() -> Self {
        LifecycleCfg {
            enforce: Enforce::Warn, // dev default; production enclave sets `block`
            passport_path: String::new(),
            ruleset_version: RULESET_VERSION.into(),
        }
    }
}

/// The pinned regulatory rule-set version. Every classification cites it.
pub const RULESET_VERSION: &str = "2026.06";
