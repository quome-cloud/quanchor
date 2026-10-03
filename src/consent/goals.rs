//! consent::goals — G4 goals-of-care / values-conflict flagger. Each patient may
//! carry a [`GoalsRecord`] of documented values (e.g. comfort-focused, DNR, no
//! blood products). An action that contradicts a recorded value is flagged and
//! escalated to human review (C4.9). Two resolution paths:
//!
//! - **scored** — when the action carries a conflict `score` in [0, 1] (a checker
//!   or model supplied it), the gate escalates at or above a clinician-tunable
//!   `threshold`. This is the E5 sweep path.
//! - **rule-based** — otherwise, the gate consults a contraindication map
//!   (action class → values it contradicts) against the patient's recorded values.

use super::{AgentAction, ConsentControl, ConsentControlFn, ConsentDecision, ConsentEffect};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// A patient's documented goals-of-care / values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalsRecord {
    pub patient: String,
    pub values: BTreeSet<String>,
}

/// The G4 checker.
#[derive(Debug, Clone)]
pub struct GoalsChecker {
    records: HashMap<String, BTreeSet<String>>,
    /// action class → the values it contradicts.
    contraindications: HashMap<String, BTreeSet<String>>,
    /// Escalate at or above this conflict score (scored path).
    pub threshold: f64,
}

impl Default for GoalsChecker {
    fn default() -> Self {
        GoalsChecker {
            records: HashMap::new(),
            contraindications: HashMap::new(),
            threshold: 0.5,
        }
    }
}

impl GoalsChecker {
    pub fn new(threshold: f64) -> Self {
        GoalsChecker {
            threshold,
            ..Default::default()
        }
    }

    pub fn insert_record(&mut self, r: GoalsRecord) -> &mut Self {
        self.records.insert(r.patient.clone(), r.values);
        self
    }

    /// Declare that `action` contradicts `value`.
    pub fn contraindicate(&mut self, action: &str, value: &str) -> &mut Self {
        self.contraindications
            .entry(action.to_string())
            .or_default()
            .insert(value.to_string());
        self
    }

    /// The recorded values this action contradicts for its patient (rule path).
    fn conflicting_values(&self, action: &AgentAction) -> BTreeSet<String> {
        let patient_values = match self.records.get(&action.patient) {
            Some(v) => v,
            None => return BTreeSet::new(),
        };
        match self.contraindications.get(&action.action) {
            Some(contra) => patient_values.intersection(contra).cloned().collect(),
            None => BTreeSet::new(),
        }
    }
}

impl ConsentControlFn for GoalsChecker {
    fn decide(&self, action: &AgentAction) -> ConsentDecision {
        // Scored path: trust the supplied conflict score against the threshold.
        if let Some(score) = action.score {
            if score >= self.threshold {
                return ConsentDecision::stop(
                    ConsentEffect::Escalate,
                    ConsentControl::GoalsOfCare,
                    format!("goals-of-care conflict score {score:.2} >= threshold {:.2}", self.threshold),
                );
            }
            return ConsentDecision::allow();
        }

        // Rule-based path: any recorded value the action contradicts.
        let conflicts = self.conflicting_values(action);
        if !conflicts.is_empty() {
            let listed = conflicts.into_iter().collect::<Vec<_>>().join(", ");
            return ConsentDecision::stop(
                ConsentEffect::Escalate,
                ConsentControl::GoalsOfCare,
                format!("action `{}` conflicts with recorded values: {listed}", action.action),
            );
        }
        ConsentDecision::allow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> GoalsChecker {
        let mut c = GoalsChecker::new(0.5);
        c.insert_record(GoalsRecord {
            patient: "p1".into(),
            values: ["comfort_focused", "DNR"].iter().map(|s| s.to_string()).collect(),
        })
        .contraindicate("order_aggressive_chemo", "comfort_focused")
        .contraindicate("order_resuscitation", "DNR");
        c
    }

    fn act(action: &str, score: Option<f64>) -> AgentAction {
        let mut a = AgentAction::new("c", "p1", action, "treatment");
        a.score = score;
        a
    }

    #[test]
    fn rule_conflict_escalates() {
        let c = checker();
        let d = c.decide(&act("order_aggressive_chemo", None));
        assert_eq!(d.effect, ConsentEffect::Escalate);
        assert!(d.reasons[0].contains("comfort_focused"));
    }

    #[test]
    fn non_conflicting_action_allowed() {
        let c = checker();
        assert_eq!(c.decide(&act("order_acetaminophen", None)).effect, ConsentEffect::Allow);
    }

    #[test]
    fn patient_without_record_passes_rule_path() {
        let c = checker();
        let mut a = act("order_aggressive_chemo", None);
        a.patient = "unknown".into();
        assert_eq!(c.decide(&a).effect, ConsentEffect::Allow);
    }

    #[test]
    fn scored_above_threshold_escalates() {
        let c = checker();
        assert_eq!(c.decide(&act("anything", Some(0.8))).effect, ConsentEffect::Escalate);
    }

    #[test]
    fn scored_below_threshold_allows() {
        let c = checker();
        assert_eq!(c.decide(&act("anything", Some(0.3))).effect, ConsentEffect::Allow);
    }

    #[test]
    fn threshold_is_tunable() {
        let mut c = checker();
        c.threshold = 0.9;
        // 0.8 now passes under the stricter threshold.
        assert_eq!(c.decide(&act("anything", Some(0.8))).effect, ConsentEffect::Allow);
    }
}
