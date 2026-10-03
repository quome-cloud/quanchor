//! The stable signal contract the scorer consumes, plus adapters that map each
//! real upstream decision into it. `Option` slots are `None` when their layer
//! did not run; `None` contributes nothing to risk (graceful degradation).
//!
//! Reconciliation note: the upstream layers expose *categorical* decisions, not
//! the idealized `f64` margins of the 009 spec. `policy::Effect` is
//! Allow/Deny/Escalate (no numeric margin); `retrieval::TrustTier` is a 3-level
//! enum. The `SignalsBuilder` adapters below collapse those into `[0,1]` risk
//! contributions per the documented mapping.

use serde::{Deserialize, Serialize};

/// Static per-tool risk class. Ordering matters: higher class ⇒ more caution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    Low = 0,
    Medium = 1,
    High = 2,
    LifeCritical = 3,
}

/// One upstream layer's reason for blocking/escalating, kept for the audit trail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenyReason {
    pub layer: String,
    pub reason: String,
}

/// Aggregated, layer-agnostic signals for one action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OversightSignals {
    pub deny_reasons: Vec<DenyReason>,
    pub max_block_confidence: f64,
    pub taint_level: Option<f64>,
    pub policy_margin: Option<f64>,
    pub provenance_trust: Option<f64>,
    pub identity_trust: Option<f64>,
    pub tool_risk: ToolRisk,
}

impl OversightSignals {
    /// An empty signal set for a tool of the given risk class.
    pub fn for_tool(tool_risk: ToolRisk) -> Self {
        OversightSignals {
            deny_reasons: Vec::new(),
            max_block_confidence: 0.0,
            taint_level: None,
            policy_margin: None,
            provenance_trust: None,
            identity_trust: None,
            tool_risk,
        }
    }
}

const TAINT_SATURATE: f64 = 4.0;

/// Builds `OversightSignals` by absorbing each real upstream decision. Each
/// `with_*` is independent so a layer that didn't run simply isn't called.
pub struct SignalsBuilder {
    s: OversightSignals,
}

impl SignalsBuilder {
    pub fn new(tool_risk: ToolRisk) -> Self {
        SignalsBuilder {
            s: OversightSignals::for_tool(tool_risk),
        }
    }

    pub fn with_policy(mut self, d: &crate::policy::PolicyDecision) -> Self {
        use crate::policy::Effect;
        self.s.policy_margin = Some(match d.effect {
            Effect::Allow => 0.0,
            Effect::Escalate => 0.5,
            Effect::Deny => 1.0,
        });
        if d.effect != Effect::Allow {
            for r in &d.reasons {
                self.s.deny_reasons.push(DenyReason {
                    layer: "policy".into(),
                    reason: r.clone(),
                });
            }
            self.s.max_block_confidence = self.s.max_block_confidence.max(match d.effect {
                Effect::Deny => 1.0,
                _ => 0.5,
            });
        }
        self
    }

    pub fn with_egress(mut self, findings: &[crate::egress::EgressFinding]) -> Self {
        self.s.taint_level = Some((findings.len() as f64 / TAINT_SATURATE).min(1.0));
        for f in findings {
            self.s.deny_reasons.push(DenyReason {
                layer: "egress".into(),
                reason: format!("{} leaked via {}", f.label, f.via),
            });
        }
        if !findings.is_empty() {
            self.s.max_block_confidence = self.s.max_block_confidence.max(0.9);
        }
        self
    }

    pub fn with_retrieval(mut self, tier: crate::retrieval::TrustTier) -> Self {
        self.s.provenance_trust = Some(1.0 - tier.weight() as f64);
        self
    }

    pub fn with_identity(mut self, d: &crate::identity::IdentityDecision) -> Self {
        use crate::identity::IdentityEffect;
        let frac_failed = if d.checks.is_empty() {
            0.0
        } else {
            d.checks.iter().filter(|c| !c.ok).count() as f64 / d.checks.len() as f64
        };
        let mut risk = match d.effect {
            IdentityEffect::Deny => 1.0,
            IdentityEffect::Allow => frac_failed,
        };
        if d.proven_principal.is_none() {
            risk = risk.max(0.5);
        }
        self.s.identity_trust = Some(risk);
        if d.effect == IdentityEffect::Deny {
            self.s.deny_reasons.push(DenyReason {
                layer: "identity".into(),
                reason: "identity check failed".into(),
            });
            self.s.max_block_confidence = self.s.max_block_confidence.max(1.0);
        }
        self
    }

    pub fn build(self) -> OversightSignals {
        self.s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_risk_orders_low_to_lifecritical() {
        assert!(ToolRisk::LifeCritical > ToolRisk::High);
        assert!(ToolRisk::High > ToolRisk::Medium);
        assert!(ToolRisk::Medium > ToolRisk::Low);
    }

    #[test]
    fn empty_signals_have_no_optional_contributions() {
        let s = OversightSignals::for_tool(ToolRisk::Low);
        assert!(s.taint_level.is_none());
        assert!(s.policy_margin.is_none());
        assert!(s.provenance_trust.is_none());
        assert!(s.identity_trust.is_none());
        assert_eq!(s.max_block_confidence, 0.0);
        assert!(s.deny_reasons.is_empty());
    }

    #[test]
    fn builder_maps_policy_egress_identity_retrieval() {
        use crate::egress::EgressFinding;
        use crate::policy::{Effect, PolicyDecision};
        use crate::retrieval::TrustTier;

        let policy = PolicyDecision {
            effect: Effect::Escalate,
            matched_rule: Some("r1".into()),
            reasons: vec!["off-formulary".into()],
        };
        let findings = vec![EgressFinding {
            prov_id: "p1".into(),
            label: "mrn".into(),
            arg_path: "note".into(),
            via: "base64".into(),
        }];

        let s = SignalsBuilder::new(ToolRisk::High)
            .with_policy(&policy)
            .with_egress(&findings)
            .with_retrieval(TrustTier::Unverified)
            .build();

        assert_eq!(s.policy_margin, Some(0.5)); // Escalate → 0.5
        assert_eq!(s.taint_level, Some(0.25)); // 1 finding / SATURATE(4)
        // 1.0 - weight(0.2); weight() is f32, so compare with tolerance.
        assert!((s.provenance_trust.unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(s.deny_reasons.len(), 2); // policy reason + egress finding
        assert!(s.identity_trust.is_none()); // identity not supplied
    }
}
