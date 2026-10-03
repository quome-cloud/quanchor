//! The static allow-list baseline: a role -> allowed-actions map, the shape of
//! traditional RBAC. It is context-blind by construction (ignores args and
//! attrs), which is exactly what the E1 over-permissioning experiment measures
//! against the context-aware engines.

use super::{Effect, PolicyDecision, PolicyEngine, Request};
use std::collections::HashMap;

pub struct StaticRbac {
    /// role -> set of allowed action names.
    allow: HashMap<String, Vec<String>>,
}

impl StaticRbac {
    pub fn new(allow: HashMap<String, Vec<String>>) -> Self {
        StaticRbac { allow }
    }

    /// A small clinical default: roles permitted a fixed set of tools, regardless
    /// of encounter/panel/dose — the over-permissioning the paper highlights.
    pub fn clinical_default() -> Self {
        let mut m = HashMap::new();
        m.insert(
            "prescriber".into(),
            vec![
                "order_medication".into(),
                "order_lab".into(),
                "view_record".into(),
            ],
        );
        m.insert(
            "nurse".into(),
            vec!["administer".into(), "view_record".into()],
        );
        m.insert("read_only".into(), vec!["view_record".into()]);
        StaticRbac::new(m)
    }
}

impl PolicyEngine for StaticRbac {
    fn decide(&self, req: &Request) -> PolicyDecision {
        match self.allow.get(&req.principal) {
            Some(actions) if actions.iter().any(|a| a == &req.action) => PolicyDecision {
                effect: Effect::Allow,
                matched_rule: Some("rbac:allow".into()),
                reasons: vec![format!("{} may {}", req.principal, req.action)],
            },
            _ => PolicyDecision::deny(format!("{} not permitted {}", req.principal, req.action)),
        }
    }
    fn name(&self) -> &'static str {
        "static_rbac"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(principal: &str, action: &str, args: serde_json::Value) -> Request {
        Request {
            principal: principal.into(),
            action: action.into(),
            resource: "patient-1".into(),
            args,
            attrs: json!({}),
        }
    }

    #[test]
    fn allows_listed_action_denies_unlisted() {
        let r = StaticRbac::clinical_default();
        assert_eq!(
            r.decide(&req("prescriber", "order_medication", json!({})))
                .effect,
            Effect::Allow
        );
        assert_eq!(
            r.decide(&req("nurse", "order_medication", json!({})))
                .effect,
            Effect::Deny
        );
        assert_eq!(
            r.decide(&req("read_only", "view_record", json!({}))).effect,
            Effect::Allow
        );
    }

    #[test]
    fn is_context_blind_overpermits() {
        // The hallmark of the baseline: it allows an opioid order with NO active
        // encounter and an out-of-range dose, because it only checks role+action.
        let r = StaticRbac::clinical_default();
        let bad = req(
            "prescriber",
            "order_medication",
            json!({"drug": "morphine", "dose_mg": 9999}),
        );
        // a context-aware engine would deny this; static RBAC allows it.
        assert_eq!(r.decide(&bad).effect, Effect::Allow);
    }
}
