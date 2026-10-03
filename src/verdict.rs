//! Verdict types shared across detectors, rules and chains.

use serde::{Deserialize, Serialize};

/// The verdict a detector node, rule, or chain can produce.
///
/// Semantics are consistent at every level: **ALLOW** means "passes / within
/// scope / clean", **BLOCK** means "violates / out of scope / attack", and
/// **ABSTAIN** means "no opinion — defer to the next node, rule, or the chain
/// default". **ERROR** is a detector failure handled by the chain's fail policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Allow,
    Block,
    Abstain,
    Error,
}

impl Verdict {
    /// Returns true if this verdict is terminal at the node level under
    /// stop-on-first-block semantics.
    pub fn is_block(self) -> bool {
        matches!(self, Verdict::Block)
    }

    /// Returns true if this verdict is terminal under stop-on-first-allow.
    pub fn is_allow(self) -> bool {
        matches!(self, Verdict::Allow)
    }

    /// A short uppercase label for terminal rendering.
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Allow => "ALLOW",
            Verdict::Block => "BLOCK",
            Verdict::Abstain => "ABSTAIN",
            Verdict::Error => "ERROR",
        }
    }
}

/// The result of a single detector node evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeVerdict {
    /// The detector kind, e.g. `"regex"`, `"entropy"`, `"judge"`.
    pub kind: String,
    /// The detector's model/version string, for reproducibility and citation.
    pub version: String,
    /// The verdict this node produced.
    pub verdict: Verdict,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
    /// Wall-clock latency of this node in milliseconds.
    pub latency_ms: f64,
    /// A one-line human rationale.
    pub rationale: String,
    /// The raw signal where one exists (entropy bits, classifier probability,
    /// similarity score), enabling ROC/AUC in the benchmark harness.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl NodeVerdict {
    /// Construct a verdict with no raw score.
    pub fn new(
        kind: impl Into<String>,
        version: impl Into<String>,
        verdict: Verdict,
        confidence: f64,
        latency_ms: f64,
        rationale: impl Into<String>,
    ) -> Self {
        NodeVerdict {
            kind: kind.into(),
            version: version.into(),
            verdict,
            confidence: confidence.clamp(0.0, 1.0),
            latency_ms,
            rationale: rationale.into(),
            score: None,
        }
    }

    /// Attach a raw score to this verdict.
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }
}
