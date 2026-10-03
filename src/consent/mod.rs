//! Consent, disclosure & participatory governance (paper 013). An *enforcing*
//! gateway layer that runs after identity (008) and policy (005): it gates an
//! agent action on the patient's recorded consent scope (G1), attaches an
//! AI-involvement disclosure to patient-facing output (G2), requires a
//! credentialed-operator attestation for credentialed actions (G3), flags actions
//! conflicting with recorded goals-of-care (G4), and anchors participatory-
//! governance inputs to the 003 audit chain (G5).
//!
//! Broken / withdrawn / expired / missing consent flips an action to `Escalate`
//! (human review via 009), never a silent allow; an explicit deny-provision is a
//! hard `Deny`. Every decision, disclosure, and attestation is auditable.
//!
//! Submodules: [`engine`] (G1 FHIR-Consent scope), [`disclosure`] (G2 detector +
//! attacher), [`competency`] (G3 attestation gate), [`goals`] (G4 conflict
//! flagger), [`ledger`] (G5 governance ledger), [`metrics`] (estimators +
//! McNemar + kappa), [`reporter`] (consent-compliance report).

use serde::{Deserialize, Serialize};

pub mod competency;
pub mod disclosure;
pub mod engine;
pub mod goals;
pub mod ledger;
pub mod metrics;
pub mod reporter;

use crate::verdict::Verdict;

/// The terminal effect the consent layer attributes to one action. `Deny` is a
/// hard block (an explicit deny-provision or an unsupported purpose); `Escalate`
/// routes to human review (broken/withdrawn/expired/missing consent, a competency
/// gap, or a goals-of-care conflict) — the layer never silently allows when
/// consent is in doubt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsentEffect {
    Allow,
    Deny,
    Escalate,
}

impl ConsentEffect {
    /// Map the consent effect onto a gateway terminal `Verdict`. `Allow` → `Allow`;
    /// both `Deny` and `Escalate` stop the action on the hot path (`Block`) — the
    /// escalation is then routed by 009, which the consent layer records but does
    /// not itself perform.
    pub fn to_verdict(self) -> Verdict {
        match self {
            ConsentEffect::Allow => Verdict::Allow,
            _ => Verdict::Block,
        }
    }

    /// True when the action was stopped on the hot path (deny or escalate).
    pub fn is_stopped(self) -> bool {
        !matches!(self, ConsentEffect::Allow)
    }
}

/// Which control was decisive for a [`ConsentDecision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsentControl {
    ScopeGate,
    Competency,
    GoalsOfCare,
    /// No control fired — the action is permitted (disclosure may still attach).
    None,
}

/// A signed attestation that a named operator holds a set of credentials. The
/// signature reuses the 008 ed25519 plumbing; the competency gate verifies it
/// before honouring the claimed credentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorAttestation {
    pub operator_id: String,
    pub credentials: std::collections::BTreeSet<String>,
    /// Hex ed25519 signature over `operator_id|credentials`, or empty if unsigned.
    #[serde(default)]
    pub signature: String,
}

/// One agent action presented to the consent layer. Carries everything the five
/// controls need: the patient (→ consent directive + goals record), the action +
/// FHIR purpose-of-use (→ scope), the operator attestation (→ competency), any
/// generated patient-facing text (→ disclosure), and an optional conflict score
/// (→ goals-of-care threshold).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentAction {
    pub case_id: String,
    pub patient: String,
    pub action: String,
    /// FHIR purpose-of-use: treatment | research | contact | marketing.
    pub purpose: String,
    #[serde(default)]
    pub operator: Option<OperatorAttestation>,
    /// Generated patient-facing text, if this action produces any (disclosure path).
    #[serde(default)]
    pub patient_facing: Option<String>,
    /// Goals-of-care conflict score in [0, 1], when a checker supplied one.
    #[serde(default)]
    pub score: Option<f64>,
}

impl AgentAction {
    /// Convenience constructor for the common scope-only case.
    pub fn new(
        case_id: impl Into<String>,
        patient: impl Into<String>,
        action: impl Into<String>,
        purpose: impl Into<String>,
    ) -> Self {
        AgentAction {
            case_id: case_id.into(),
            patient: patient.into(),
            action: action.into(),
            purpose: purpose.into(),
            operator: None,
            patient_facing: None,
            score: None,
        }
    }
}

/// An explainable consent decision: the effect, which control decided it, the
/// human-readable reasons, and (on the output path) the disclosure string that
/// was attached.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsentDecision {
    pub effect: ConsentEffect,
    pub control: ConsentControl,
    pub reasons: Vec<String>,
    #[serde(default)]
    pub disclosure: Option<String>,
}

impl ConsentDecision {
    pub fn allow() -> Self {
        ConsentDecision {
            effect: ConsentEffect::Allow,
            control: ConsentControl::None,
            reasons: vec![],
            disclosure: None,
        }
    }

    pub fn stop(effect: ConsentEffect, control: ConsentControl, reason: impl Into<String>) -> Self {
        ConsentDecision {
            effect,
            control,
            reasons: vec![reason.into()],
            disclosure: None,
        }
    }
}

/// A control that decides on a single action. Each of G1/G3/G4 implements this.
pub trait ConsentControlFn {
    fn decide(&self, action: &AgentAction) -> ConsentDecision;
}

/// The composed consent layer: scope (G1) → competency (G3) → goals-of-care (G4),
/// short-circuiting on the first non-`Allow`; the disclosure stage (G2) always
/// runs on `patient_facing` output and is recorded even when the action is
/// allowed.
pub struct ConsentLayer {
    pub scope: engine::ScopeGate,
    pub competency: competency::CompetencyGate,
    pub goals: goals::GoalsChecker,
    pub disclosure: disclosure::DisclosureAttacher,
}

impl ConsentLayer {
    pub fn decide(&self, action: &AgentAction) -> ConsentDecision {
        // Controls run in order; the first to stop the action wins.
        for control in [
            &self.scope as &dyn ConsentControlFn,
            &self.competency as &dyn ConsentControlFn,
            &self.goals as &dyn ConsentControlFn,
        ] {
            let d = control.decide(action);
            if d.effect.is_stopped() {
                return self.with_disclosure(action, d);
            }
        }
        self.with_disclosure(action, ConsentDecision::allow())
    }

    /// Attach a disclosure string to any patient-facing output that lacks one.
    fn with_disclosure(&self, action: &AgentAction, mut d: ConsentDecision) -> ConsentDecision {
        if let Some(text) = &action.patient_facing {
            d.disclosure = Some(self.disclosure.ensure(text));
        }
        d
    }
}

/// Serialize a consent decision for the 003 audit chain (mirrors
/// `identity::audit_body`).
pub fn audit_body(action: &AgentAction, d: &ConsentDecision) -> String {
    serde_json::json!({
        "layer": "consent",
        "case_id": action.case_id,
        "patient": action.patient,
        "action": action.action,
        "purpose": action.purpose,
        "effect": d.effect,
        "control": d.control,
        "reasons": d.reasons,
        "disclosed": d.disclosure.is_some(),
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_maps_to_verdict() {
        assert_eq!(ConsentEffect::Allow.to_verdict(), Verdict::Allow);
        assert_eq!(ConsentEffect::Deny.to_verdict(), Verdict::Block);
        assert_eq!(ConsentEffect::Escalate.to_verdict(), Verdict::Block);
    }

    #[test]
    fn stopped_effects() {
        assert!(ConsentEffect::Deny.is_stopped());
        assert!(ConsentEffect::Escalate.is_stopped());
        assert!(!ConsentEffect::Allow.is_stopped());
    }

    #[test]
    fn effect_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&ConsentEffect::Escalate).unwrap(), "\"escalate\"");
    }

    #[test]
    fn agent_action_round_trips() {
        let a = AgentAction::new("c1", "p1", "send_reminder", "contact");
        let json = serde_json::to_string(&a).unwrap();
        let back: AgentAction = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn audit_body_is_consent_layer() {
        let a = AgentAction::new("c1", "p1", "share", "research");
        let d = ConsentDecision::stop(ConsentEffect::Escalate, ConsentControl::ScopeGate, "withdrawn");
        let v: serde_json::Value = serde_json::from_str(&audit_body(&a, &d)).unwrap();
        assert_eq!(v["layer"], "consent");
        assert_eq!(v["effect"], "escalate");
        assert_eq!(v["control"], "scope_gate");
    }
}
