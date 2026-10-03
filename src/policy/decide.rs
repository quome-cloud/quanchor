//! The authorization flow: run an engine, map its effect to a gateway outcome,
//! and log an Authorization entry to the 003 audit chain. Escalate (break-glass)
//! maps to an audited Block with a loud reason until paper 006 routes to a human.
//! Fail-closed: an engine that errors on a safety-critical action denies.

use super::{Effect, PolicyDecision, PolicyEngine, Request};
use crate::audit::AuditSink;
use crate::Result;

/// The gateway-facing outcome of authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Allow,
    Block,
}

/// Map a policy effect to a gateway outcome. Allow->Allow; Deny->Block;
/// Escalate->Block (break-glass; real human routing is paper 006).
pub fn effect_to_outcome(effect: Effect) -> Outcome {
    match effect {
        Effect::Allow => Outcome::Allow,
        Effect::Deny | Effect::Escalate => Outcome::Block,
    }
}

/// Authorize a request: decide, map, and log. `policy_version` is the verified
/// bundle version (for traceability). Returns (outcome, decision).
pub fn authorize(
    engine: &dyn PolicyEngine,
    req: &Request,
    policy_version: &str,
    audit: &AuditSink,
) -> Result<(Outcome, PolicyDecision)> {
    let decision = engine.decide(req);
    let outcome = effect_to_outcome(decision.effect);
    let reason = if decision.effect == Effect::Escalate {
        "break-glass: escalate".to_string()
    } else {
        decision.reasons.first().cloned().unwrap_or_default()
    };
    let body = serde_json::json!({
        "event": "authorization",
        "engine": engine.name(),
        "policy_version": policy_version,
        "principal": req.principal,
        "action": req.action,
        "resource": req.resource,
        "effect": decision.effect,
        "matched_rule": decision.matched_rule,
        "reason": reason,
    });
    audit.append_authorization_json(body.to_string())?;
    Ok((outcome, decision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::chain::{parse_line, EntryKind};
    use crate::audit::store::{Mode, StoreCfg, TamperEvidentLog};
    use crate::audit::AuditSink;
    use crate::policy::rbac::StaticRbac;
    use serde_json::json;

    fn req(principal: &str, action: &str) -> Request {
        Request {
            principal: principal.into(),
            action: action.into(),
            resource: "patient-1".into(),
            args: json!({}),
            attrs: json!({}),
        }
    }

    fn chained_sink(dir: &std::path::Path) -> AuditSink {
        AuditSink::Chained(
            TamperEvidentLog::open(StoreCfg {
                path: dir.join("audit.jsonl"),
                mode: Mode::Chained,
                signer: None,
                anchor: None,
                batch: 4,
                fail_open: false,
            })
            .unwrap(),
        )
    }

    #[test]
    fn effect_mapping() {
        assert_eq!(effect_to_outcome(Effect::Allow), Outcome::Allow);
        assert_eq!(effect_to_outcome(Effect::Deny), Outcome::Block);
        assert_eq!(effect_to_outcome(Effect::Escalate), Outcome::Block);
    }

    #[test]
    fn authorize_logs_decision_to_chain() {
        let dir = tempfile::tempdir().unwrap();
        let sink = chained_sink(dir.path());
        let eng = StaticRbac::clinical_default();
        let (out, dec) = authorize(&eng, &req("nurse", "order_medication"), "v1", &sink).unwrap();
        assert_eq!(out, Outcome::Block);
        assert_eq!(dec.effect, Effect::Deny);
        drop(sink);
        let text = std::fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
        let e = parse_line(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(e.kind, EntryKind::Authorization);
        assert_eq!(e.body["engine"], "static_rbac");
        assert_eq!(e.body["effect"], "deny");
        assert_eq!(e.body["policy_version"], "v1");
    }
}
