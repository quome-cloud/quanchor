//! consent::reporter — the consolidated consent-compliance report. Accumulates
//! per-control outcomes from a decision stream and serializes a JSON summary
//! (prevention counts, disclosure coverage, escalation cost) with 95% Wilson CIs,
//! suitable for the 003 audit log and the C4 governance record.

use super::metrics::{rate, wilson_ci};
use super::{ConsentControl, ConsentDecision, ConsentEffect};
use serde::Serialize;
use serde_json::{json, Value};

/// Running tallies over a consent-decision stream.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ConsentReport {
    pub total: u64,
    pub allowed: u64,
    pub denied: u64,
    pub escalated: u64,
    /// Stops attributed to each control.
    pub by_scope: u64,
    pub by_competency: u64,
    pub by_goals: u64,
    /// Patient-facing outputs seen, and how many carried a disclosure.
    pub patient_facing: u64,
    pub disclosed: u64,
}

impl ConsentReport {
    pub fn new() -> Self {
        ConsentReport::default()
    }

    /// Fold one decision into the report.
    pub fn observe(&mut self, d: &ConsentDecision) {
        self.total += 1;
        match d.effect {
            ConsentEffect::Allow => self.allowed += 1,
            ConsentEffect::Deny => self.denied += 1,
            ConsentEffect::Escalate => self.escalated += 1,
        }
        if d.effect.is_stopped() {
            match d.control {
                ConsentControl::ScopeGate => self.by_scope += 1,
                ConsentControl::Competency => self.by_competency += 1,
                ConsentControl::GoalsOfCare => self.by_goals += 1,
                ConsentControl::None => {}
            }
        }
        if d.disclosure.is_some() {
            self.patient_facing += 1;
            self.disclosed += 1;
        }
    }

    /// Record a patient-facing output that did *not* receive a disclosure (used
    /// when measuring the un-gated baseline).
    pub fn observe_undisclosed_patient_facing(&mut self) {
        self.patient_facing += 1;
    }

    /// Disclosure coverage = disclosed / patient_facing.
    pub fn disclosure_coverage(&self) -> f64 {
        rate(self.disclosed, self.patient_facing)
    }

    /// Escalation rate = escalated / total.
    pub fn escalation_rate(&self) -> f64 {
        rate(self.escalated, self.total)
    }

    /// Serialize the report with 95% Wilson CIs on the headline rates.
    pub fn to_json(&self) -> Value {
        let (dc_lo, dc_hi) = wilson_ci(self.disclosed, self.patient_facing, 1.96);
        let (esc_lo, esc_hi) = wilson_ci(self.escalated, self.total, 1.96);
        json!({
            "total": self.total,
            "allowed": self.allowed,
            "denied": self.denied,
            "escalated": self.escalated,
            "by_control": {
                "scope_gate": self.by_scope,
                "competency": self.by_competency,
                "goals_of_care": self.by_goals,
            },
            "disclosure": {
                "patient_facing": self.patient_facing,
                "disclosed": self.disclosed,
                "coverage": self.disclosure_coverage(),
                "coverage_ci": [dc_lo, dc_hi],
            },
            "escalation": {
                "rate": self.escalation_rate(),
                "rate_ci": [esc_lo, esc_hi],
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::{ConsentControl, ConsentDecision, ConsentEffect};

    fn dec(effect: ConsentEffect, control: ConsentControl, disclosed: bool) -> ConsentDecision {
        ConsentDecision {
            effect,
            control,
            reasons: vec![],
            disclosure: if disclosed { Some("d".into()) } else { None },
        }
    }

    #[test]
    fn counts_by_effect_and_control() {
        let mut r = ConsentReport::new();
        r.observe(&dec(ConsentEffect::Allow, ConsentControl::None, true));
        r.observe(&dec(ConsentEffect::Escalate, ConsentControl::ScopeGate, false));
        r.observe(&dec(ConsentEffect::Deny, ConsentControl::ScopeGate, false));
        r.observe(&dec(ConsentEffect::Escalate, ConsentControl::Competency, false));
        assert_eq!(r.total, 4);
        assert_eq!(r.allowed, 1);
        assert_eq!(r.denied, 1);
        assert_eq!(r.escalated, 2);
        assert_eq!(r.by_scope, 2);
        assert_eq!(r.by_competency, 1);
    }

    #[test]
    fn disclosure_coverage_tracks_both_paths() {
        let mut r = ConsentReport::new();
        r.observe(&dec(ConsentEffect::Allow, ConsentControl::None, true));
        r.observe_undisclosed_patient_facing();
        assert_eq!(r.patient_facing, 2);
        assert_eq!(r.disclosed, 1);
        assert!((r.disclosure_coverage() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn json_has_cis() {
        let mut r = ConsentReport::new();
        r.observe(&dec(ConsentEffect::Allow, ConsentControl::None, true));
        let v = r.to_json();
        assert!(v["disclosure"]["coverage_ci"].is_array());
        assert!(v["escalation"]["rate_ci"].is_array());
    }
}
