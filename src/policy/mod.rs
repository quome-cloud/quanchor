//! Paper 005 — policy-as-code authorization. A PolicyEngine decides whether a
//! principal may take an action on a resource given request arguments and PIP
//! context attributes. Decisions are explainable (matched rule + reasons) and
//! logged to the 003 audit chain. Heavy engines (Cedar, Rego) are behind the
//! `policy` cargo feature; a native StaticRbac baseline always compiles.

pub mod bundle;
pub mod decide;
pub mod pip;
pub mod rbac;

#[cfg(feature = "policy")]
pub mod cedar;
#[cfg(feature = "policy")]
pub mod rego;

use serde::{Deserialize, Serialize};

/// The decision effect. Escalate (break-glass) maps to an audited Block until
/// paper 006 adds human routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Allow,
    Deny,
    Escalate,
}

/// An explainable authorization decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub effect: Effect,
    /// The id of the policy/rule that decided, when the engine reports one.
    pub matched_rule: Option<String>,
    pub reasons: Vec<String>,
}

impl PolicyDecision {
    pub fn deny(reason: impl Into<String>) -> Self {
        PolicyDecision {
            effect: Effect::Deny,
            matched_rule: None,
            reasons: vec![reason.into()],
        }
    }
    pub fn allow(matched_rule: Option<String>) -> Self {
        PolicyDecision {
            effect: Effect::Allow,
            matched_rule,
            reasons: Vec::new(),
        }
    }
}

/// An authorization request: who, what, on what, with which arguments and context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub principal: String,        // role/identity (from header; paper 008 later)
    pub action: String,           // tool name, e.g. "order_medication"
    pub resource: String,         // e.g. a patient ref
    pub args: serde_json::Value,  // tool arguments (dose, patient_id, ...) -> P4
    pub attrs: serde_json::Value, // PIP context (encounter/panel/formulary/...)
}

/// A pluggable policy engine.
pub trait PolicyEngine {
    fn decide(&self, req: &Request) -> PolicyDecision;
    fn name(&self) -> &'static str; // "static_rbac" | "cedar" | "rego"
}

/// Policy configuration. This is the config surface for the gateway request-path
/// integration, which is deferred (see the spec's Out-of-scope); the fields are
/// consumed when the policy layer is wired inline. `fail_closed` is the intended
/// per-action override; today every engine fails closed (deny) on error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicyCfg {
    /// "static_rbac" | "cedar" | "rego"
    pub engine: String,
    /// Path to the signed policy bundle directory ("" disables authorization).
    pub bundle_dir: String,
    /// Path to the synthetic PIP attribute store ("" = empty attributes).
    pub attrs_path: String,
    /// Fail closed (deny) on engine error for safety-critical actions.
    pub fail_closed: bool,
}

impl Default for PolicyCfg {
    fn default() -> Self {
        PolicyCfg {
            engine: "static_rbac".into(),
            bundle_dir: String::new(),
            attrs_path: String::new(),
            fail_closed: true,
        }
    }
}

pub use decide::authorize;
pub use decide::{effect_to_outcome, Outcome};
