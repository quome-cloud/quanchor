//! Machine-readable Predetermined Change Control Plan (PCCP). A PCCP pre-authorises
//! a bounded envelope of future changes to a cleared agent: which component classes
//! may change, and by how much. `evaluate` judges a set of `ComponentDelta`s (from
//! [`super::fingerprint::diff`]) against the plan and returns an in-/out-of-envelope
//! verdict — the L3 decision. **Fail-closed:** a change to a component with no rule,
//! or a bounded change whose magnitude is unknown, is treated as out-of-envelope.

use super::fingerprint::{Component, ComponentDelta};
use serde::{Deserialize, Serialize};

/// What the plan pre-authorises for one component class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "policy")]
pub enum ChangePolicy {
    /// No change permitted without re-review (e.g. a new tool that adds capability).
    Forbidden,
    /// Any change permitted (e.g. a prompt copy-edit within tagged sections).
    AnyChange,
    /// Change permitted only up to a magnitude bound in [0,1]; unknown magnitude
    /// is out-of-envelope (fail-closed).
    BoundedMagnitude { max_fraction: f64 },
}

/// One per-component clause of the plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentRule {
    pub component: Component,
    #[serde(flatten)]
    pub change: ChangePolicy,
}

/// The pre-authorised change envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pccp {
    pub allowed: Vec<ComponentRule>,
}

impl Pccp {
    fn policy_for(&self, c: Component) -> Option<&ChangePolicy> {
        self.allowed.iter().find(|r| r.component == c).map(|r| &r.change)
    }
}

/// The verdict of evaluating a change against a PCCP.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeVerdict {
    pub in_envelope: bool,
    /// Components whose change exceeded the envelope (empty iff `in_envelope`).
    pub violations: Vec<Component>,
}

/// Judge a set of component deltas against the plan (fail-closed).
pub fn evaluate(deltas: &[ComponentDelta], pccp: &Pccp) -> EnvelopeVerdict {
    let mut violations = Vec::new();
    for d in deltas {
        let ok = match pccp.policy_for(d.component) {
            None => false,                       // unlisted ⇒ forbidden by default
            Some(ChangePolicy::Forbidden) => false,
            Some(ChangePolicy::AnyChange) => true,
            Some(ChangePolicy::BoundedMagnitude { max_fraction }) => {
                matches!(d.magnitude, Some(m) if m <= *max_fraction)
            }
        };
        if !ok {
            violations.push(d.component);
        }
    }
    EnvelopeVerdict { in_envelope: violations.is_empty(), violations }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(c: Component, magnitude: Option<f64>) -> ComponentDelta {
        ComponentDelta { component: c, from: "a".into(), to: "b".into(), magnitude }
    }

    #[test]
    fn in_envelope_change_accepted() {
        let pccp = Pccp {
            allowed: vec![ComponentRule { component: Component::Prompt, change: ChangePolicy::AnyChange }],
        };
        let v = evaluate(&[delta(Component::Prompt, None)], &pccp);
        assert!(v.in_envelope);
        assert!(v.violations.is_empty());
    }

    #[test]
    fn forbidden_component_rejected() {
        let pccp = Pccp {
            allowed: vec![ComponentRule { component: Component::Weights, change: ChangePolicy::Forbidden }],
        };
        let v = evaluate(&[delta(Component::Weights, Some(0.001))], &pccp);
        assert!(!v.in_envelope);
        assert_eq!(v.violations, vec![Component::Weights]);
    }

    #[test]
    fn unlisted_component_is_forbidden_by_default() {
        let pccp = Pccp { allowed: vec![] };
        let v = evaluate(&[delta(Component::Tools, None)], &pccp);
        assert!(!v.in_envelope);
        assert_eq!(v.violations, vec![Component::Tools]);
    }

    #[test]
    fn bounded_within_bound_ok_over_bound_violation() {
        let pccp = Pccp {
            allowed: vec![ComponentRule {
                component: Component::Weights,
                change: ChangePolicy::BoundedMagnitude { max_fraction: 0.05 },
            }],
        };
        assert!(evaluate(&[delta(Component::Weights, Some(0.04))], &pccp).in_envelope);
        assert!(!evaluate(&[delta(Component::Weights, Some(0.06))], &pccp).in_envelope);
        // Unknown magnitude under a bounded policy is out-of-envelope (fail-closed).
        assert!(!evaluate(&[delta(Component::Weights, None)], &pccp).in_envelope);
    }
}
