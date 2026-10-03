//! consent::engine — G1 consent-scope resolution over a FHIR-Consent-shaped
//! directive store. Default-deny: an action is permitted only when the patient's
//! consent is `Active` and a matching provision permits the action's purpose.
//! Withdrawn / expired / draft / missing consent escalates to human review; an
//! explicit deny-provision is a hard deny.

use super::{AgentAction, ConsentControl, ConsentControlFn, ConsentDecision, ConsentEffect};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lifecycle status of a consent directive (subset of FHIR `Consent.status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsentStatus {
    Active,
    Withdrawn,
    Expired,
    Draft,
}

/// Whether a provision permits or denies the actions it matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProvisionType {
    Permit,
    Deny,
}

/// A FHIR-Consent `provision`: permit or deny a set of actions for a purpose.
/// An empty `actions` list matches every action under that purpose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provision {
    pub kind: ProvisionType,
    pub purpose: String,
    #[serde(default)]
    pub actions: Vec<String>,
}

impl Provision {
    /// Does this provision match the action's purpose and action class? An empty
    /// `actions` list matches any action under the purpose; a non-empty one is a
    /// more specific match.
    fn matches(&self, purpose: &str, action: &str) -> Match {
        if self.purpose != purpose {
            return Match::No;
        }
        if self.actions.is_empty() {
            Match::Purpose
        } else if self.actions.iter().any(|a| a == action) {
            Match::Specific
        } else {
            Match::No
        }
    }
}

/// How specifically a provision matched — used to pick the most specific rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Match {
    No,
    Purpose,
    Specific,
}

/// One patient's consent directive: a base stance plus purpose-scoped exceptions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsentDirective {
    pub patient: String,
    pub status: ConsentStatus,
    /// The default stance when no exception matches.
    pub base: ProvisionType,
    #[serde(default)]
    pub exceptions: Vec<Provision>,
}

/// The G1 scope gate: holds a directive per patient and resolves each action.
#[derive(Debug, Clone, Default)]
pub struct ScopeGate {
    directives: HashMap<String, ConsentDirective>,
}

impl ScopeGate {
    pub fn new() -> Self {
        ScopeGate {
            directives: HashMap::new(),
        }
    }

    pub fn insert(&mut self, d: ConsentDirective) {
        self.directives.insert(d.patient.clone(), d);
    }

    pub fn from_directives(ds: impl IntoIterator<Item = ConsentDirective>) -> Self {
        let mut g = ScopeGate::new();
        for d in ds {
            g.insert(d);
        }
        g
    }
}

impl ConsentControlFn for ScopeGate {
    fn decide(&self, action: &AgentAction) -> ConsentDecision {
        let directive = match self.directives.get(&action.patient) {
            Some(d) => d,
            None => {
                return ConsentDecision::stop(
                    ConsentEffect::Escalate,
                    ConsentControl::ScopeGate,
                    format!("no consent directive on file for patient {}", action.patient),
                )
            }
        };

        // A non-active directive cannot authorize anything → human review.
        match directive.status {
            ConsentStatus::Withdrawn => {
                return ConsentDecision::stop(
                    ConsentEffect::Escalate,
                    ConsentControl::ScopeGate,
                    "consent withdrawn",
                )
            }
            ConsentStatus::Expired => {
                return ConsentDecision::stop(
                    ConsentEffect::Escalate,
                    ConsentControl::ScopeGate,
                    "consent expired",
                )
            }
            ConsentStatus::Draft => {
                return ConsentDecision::stop(
                    ConsentEffect::Escalate,
                    ConsentControl::ScopeGate,
                    "consent not yet active (draft)",
                )
            }
            ConsentStatus::Active => {}
        }

        // Most-specific matching exception wins; ties favour an explicit Deny.
        let mut best: Option<&Provision> = None;
        let mut best_match = Match::No;
        for prov in &directive.exceptions {
            let m = prov.matches(&action.purpose, &action.action);
            if m == Match::No {
                continue;
            }
            let better = m > best_match
                || (m == best_match
                    && prov.kind == ProvisionType::Deny
                    && best.map(|b| b.kind) == Some(ProvisionType::Permit));
            if better {
                best = Some(prov);
                best_match = m;
            }
        }

        match best {
            Some(p) if p.kind == ProvisionType::Deny => ConsentDecision::stop(
                ConsentEffect::Deny,
                ConsentControl::ScopeGate,
                format!("explicit deny-provision for {}/{}", action.purpose, action.action),
            ),
            Some(_) => ConsentDecision::allow(),
            None => match directive.base {
                ProvisionType::Permit => ConsentDecision::allow(),
                ProvisionType::Deny => ConsentDecision::stop(
                    ConsentEffect::Escalate,
                    ConsentControl::ScopeGate,
                    format!("no permitting provision for purpose `{}`", action.purpose),
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directive(status: ConsentStatus, base: ProvisionType, exc: Vec<Provision>) -> ConsentDirective {
        ConsentDirective {
            patient: "p1".into(),
            status,
            base,
            exceptions: exc,
        }
    }

    fn permit(purpose: &str, actions: &[&str]) -> Provision {
        Provision {
            kind: ProvisionType::Permit,
            purpose: purpose.into(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
        }
    }
    fn deny(purpose: &str, actions: &[&str]) -> Provision {
        Provision {
            kind: ProvisionType::Deny,
            purpose: purpose.into(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn act(action: &str, purpose: &str) -> AgentAction {
        AgentAction::new("c", "p1", action, purpose)
    }

    #[test]
    fn in_scope_treatment_is_allowed() {
        let g = ScopeGate::from_directives([directive(
            ConsentStatus::Active,
            ProvisionType::Deny,
            vec![permit("treatment", &[])],
        )]);
        assert_eq!(g.decide(&act("order_medication", "treatment")).effect, ConsentEffect::Allow);
    }

    #[test]
    fn out_of_scope_purpose_escalates() {
        let g = ScopeGate::from_directives([directive(
            ConsentStatus::Active,
            ProvisionType::Deny,
            vec![permit("treatment", &[])],
        )]);
        // Research is not permitted by any provision and base is deny → escalate.
        assert_eq!(g.decide(&act("share_record", "research")).effect, ConsentEffect::Escalate);
    }

    #[test]
    fn explicit_deny_is_hard_deny() {
        let g = ScopeGate::from_directives([directive(
            ConsentStatus::Active,
            ProvisionType::Permit,
            vec![deny("marketing", &[])],
        )]);
        assert_eq!(g.decide(&act("send_promo", "marketing")).effect, ConsentEffect::Deny);
    }

    #[test]
    fn specific_deny_overrides_purpose_permit() {
        let g = ScopeGate::from_directives([directive(
            ConsentStatus::Active,
            ProvisionType::Deny,
            vec![permit("treatment", &[]), deny("treatment", &["order_opioid"])],
        )]);
        assert_eq!(g.decide(&act("order_opioid", "treatment")).effect, ConsentEffect::Deny);
        // A different treatment action still rides the purpose-level permit.
        assert_eq!(g.decide(&act("order_acetaminophen", "treatment")).effect, ConsentEffect::Allow);
    }

    #[test]
    fn withdrawn_consent_escalates() {
        let g = ScopeGate::from_directives([directive(
            ConsentStatus::Withdrawn,
            ProvisionType::Permit,
            vec![permit("treatment", &[])],
        )]);
        let d = g.decide(&act("order_medication", "treatment"));
        assert_eq!(d.effect, ConsentEffect::Escalate);
        assert!(d.reasons[0].contains("withdrawn"));
    }

    #[test]
    fn expired_consent_escalates() {
        let g = ScopeGate::from_directives([directive(
            ConsentStatus::Expired,
            ProvisionType::Permit,
            vec![permit("treatment", &[])],
        )]);
        assert_eq!(g.decide(&act("order_medication", "treatment")).effect, ConsentEffect::Escalate);
    }

    #[test]
    fn missing_directive_escalates() {
        let g = ScopeGate::new();
        let d = g.decide(&act("order_medication", "treatment"));
        assert_eq!(d.effect, ConsentEffect::Escalate);
        assert!(d.reasons[0].contains("no consent directive"));
    }
}
