//! Equity-aware enforcement (paper 010). A passive, out-of-band fairness monitor
//! over a group-labeled decision stream. It does NOT block the hot path: it
//! consumes decisions (each pairing a gateway `Verdict`/policy `Effect` with the
//! protected attributes the cohort supplies — the live `AuditRecord` carries no
//! demographics) and reports subgroup disparities, fits equity-calibrated
//! mitigations, flags vulnerable cases, and emits the C7.8 equity report.
//!
//! Submodules: [`metrics`] (estimators + Wilson CI + permutation p-value),
//! [`monitor`] (streaming subgroup monitor + alerts), [`calibrator`] (mitigation
//! arms), [`flagger`] (vulnerable-population protective routing), [`reporter`]
//! (the regulatory equity report).

use std::collections::BTreeMap;

pub mod calibrator;
pub mod flagger;
pub mod metrics;
pub mod monitor;
pub mod reporter;

use crate::verdict::Verdict;

/// The terminal enforcement action attributed to one decision. The gateway emits
/// `Verdict::{Allow,Block}`; the 005 policy layer distinguishes `Effect::Escalate`
/// (break-glass / human routing) from a hard `Deny`. The equity monitor tracks all
/// three because escalation disparity is itself a fairness harm (alarm burden).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Allow,
    Deny,
    Escalate,
}

impl Action {
    /// Map a gateway terminal `Verdict` to an equity `Action`. `Allow` → `Allow`;
    /// every blocking verdict (`Block`/`Abstain`/`Error` under fail-closed) → `Deny`.
    /// Escalation is not a `Verdict`; it arrives via the 005 policy path and is
    /// constructed directly (see [`Action::Escalate`]).
    pub fn from_verdict(v: Verdict) -> Action {
        match v {
            Verdict::Allow => Action::Allow,
            _ => Action::Deny,
        }
    }

    /// True when the action restricts the agent (deny or escalate) — i.e. the
    /// "selected for enforcement" event used by demographic-parity metrics.
    pub fn is_enforced(self) -> bool {
        matches!(self, Action::Deny | Action::Escalate)
    }
}

/// One group-labeled decision in the stream the monitor consumes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecisionRecord {
    pub case_id: String,
    /// Protected attributes for this case: axis → value (e.g. "race" → "Black").
    pub group: BTreeMap<String, String>,
    /// The gateway's terminal enforcement action.
    pub action: Action,
    /// Whether `action` matched the "fair" reference decision for this case.
    #[serde(default)]
    pub correct: bool,
    /// Ground truth: the action *should* be an enforcement (deny/escalate).
    #[serde(default)]
    pub label: bool,
    /// Detector confidence, when available — used for threshold-sweep mitigations.
    #[serde(default)]
    pub score: Option<f64>,
    /// Vulnerable-population tag, if this case is one (B3).
    #[serde(default)]
    pub vulnerable: Option<String>,
}

impl DecisionRecord {
    /// Convenience constructor for a single-axis record (the common test/E1 case).
    pub fn single(case_id: impl Into<String>, axis: &str, value: &str, action: Action) -> Self {
        let mut group = BTreeMap::new();
        group.insert(axis.to_string(), value.to_string());
        DecisionRecord {
            case_id: case_id.into(),
            group,
            action,
            correct: false,
            label: false,
            score: None,
            vulnerable: None,
        }
    }

    /// The value of `axis` for this record, if labeled with it.
    pub fn value(&self, axis: &str) -> Option<&str> {
        self.group.get(axis).map(|s| s.as_str())
    }
}

/// Which fairness metric a disparity was computed under. Reported side by side
/// (impossibility theorem): no single metric is privileged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FairnessMetric {
    /// Demographic-parity difference: max pairwise enforcement-rate gap.
    DemographicParity,
    /// Equalized-odds difference: max pairwise (TPR gap + FPR gap) vs `label`.
    EqualizedOdds,
    /// Predictive-parity difference: max pairwise precision gap.
    PredictiveParity,
}

/// A measured disparity for one axis under one metric: the worst-off pair, the
/// gap, its 95% Wilson CI, and a permutation-test p-value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Disparity {
    pub axis: String,
    pub metric: FairnessMetric,
    pub group_a: String,
    pub group_b: String,
    pub gap: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub p_value: f64,
    pub n: usize,
}

/// Anything that can consume a decision stream and report per-axis disparities.
pub trait SubgroupMonitor {
    /// Observe one decision.
    fn observe(&mut self, rec: &DecisionRecord);
    /// Per-metric disparities for `axis`, computed over everything observed so far.
    fn disparities(&self, axis: &str) -> Vec<Disparity>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verdict::Verdict;

    #[test]
    fn action_maps_from_verdict() {
        assert_eq!(Action::from_verdict(Verdict::Allow), Action::Allow);
        assert_eq!(Action::from_verdict(Verdict::Block), Action::Deny);
        assert_eq!(Action::from_verdict(Verdict::Abstain), Action::Deny);
        assert_eq!(Action::from_verdict(Verdict::Error), Action::Deny);
    }

    #[test]
    fn enforced_actions_are_deny_and_escalate() {
        assert!(Action::Deny.is_enforced());
        assert!(Action::Escalate.is_enforced());
        assert!(!Action::Allow.is_enforced());
    }

    #[test]
    fn decision_record_round_trips_through_json() {
        let rec = DecisionRecord::single("c-1", "race", "Black", Action::Deny);
        let json = serde_json::to_string(&rec).unwrap();
        let back: DecisionRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(rec, back);
        assert_eq!(back.value("race"), Some("Black"));
        assert_eq!(back.value("sex"), None);
    }

    #[test]
    fn action_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Action::Escalate).unwrap(), "\"escalate\"");
    }
}
