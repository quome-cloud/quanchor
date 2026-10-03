//! EgressGuard (paper 006): the enforcement layer. Runs egress detectors over an outbound call,
//! surfaces findings to the 005 policy engine via Request.attrs.egress, and enforces the returned
//! effect — Deny -> Block, Allow (with findings) -> redact tainted spans, Escalate -> human-route
//! (the path 005 deferred to this paper). Redaction is best-effort over verbatim occurrences;
//! residual-encoding leakage is a stated limitation.

use super::taint::TaintStore;
use super::EgressFinding;
use crate::policy::{Effect, PolicyEngine, Request};
use serde_json::json;

/// What the gateway should do with an outbound call after egress evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgressOutcome {
    Allow,
    Redacted(Vec<(String, String)>),
    Block,
    Escalate,
}

pub struct EgressGuard;

impl EgressGuard {
    /// Evaluate one outbound call. `args` are (arg_path, value) pairs; `taint` is the session store.
    pub fn evaluate(
        principal: &str,
        action: &str,
        resource: &str,
        args: &[(String, String)],
        taint: &TaintStore,
        engine: &dyn PolicyEngine,
    ) -> EgressOutcome {
        let findings: Vec<EgressFinding> = taint.scan_call(args);
        let labels: Vec<&str> = findings.iter().map(|f| f.label.as_str()).collect();
        let args_map: serde_json::Map<String, serde_json::Value> = args
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        let req = Request {
            principal: principal.to_string(),
            action: action.to_string(),
            resource: resource.to_string(),
            args: serde_json::Value::Object(args_map),
            attrs: json!({"egress": {
                "phi_leaked": !findings.is_empty(),
                "labels": labels,
                "n_findings": findings.len(),
            }}),
        };
        match engine.decide(&req).effect {
            Effect::Deny => EgressOutcome::Block,
            Effect::Escalate => EgressOutcome::Escalate,
            Effect::Allow => {
                if findings.is_empty() {
                    EgressOutcome::Allow
                } else {
                    EgressOutcome::Redacted(redact(args, taint))
                }
            }
        }
    }
}

/// Replace verbatim occurrences of tagged values in the args with a [REDACTED:<label>] marker.
fn redact(args: &[(String, String)], taint: &TaintStore) -> Vec<(String, String)> {
    args.iter()
        .map(|(k, v)| {
            let mut out = v.clone();
            for (value, label) in taint.tagged_values() {
                if value.len() >= 3 && out.contains(&value) {
                    out = out.replace(&value, &format!("[REDACTED:{label}]"));
                }
            }
            (k.clone(), out)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyDecision;

    struct FixedEngine(Effect);
    impl PolicyEngine for FixedEngine {
        fn decide(&self, _req: &Request) -> PolicyDecision {
            PolicyDecision { effect: self.0, matched_rule: None, reasons: vec![] }
        }
        fn name(&self) -> &'static str {
            "fixed_test"
        }
    }

    fn store() -> TaintStore {
        let mut s = TaintStore::new();
        s.tag("123-45-6789", "p-ssn", "ssn");
        s
    }
    fn leaking() -> Vec<(String, String)> {
        vec![("body".into(), "ssn 123-45-6789 here".into())]
    }

    #[test]
    fn deny_blocks() {
        assert_eq!(
            EgressGuard::evaluate("r", "a", "res", &leaking(), &store(), &FixedEngine(Effect::Deny)),
            EgressOutcome::Block
        );
    }
    #[test]
    fn allow_with_findings_redacts() {
        match EgressGuard::evaluate("r", "a", "res", &leaking(), &store(), &FixedEngine(Effect::Allow)) {
            EgressOutcome::Redacted(a) => {
                assert!(!a[0].1.contains("123-45-6789"), "tainted value removed");
                assert!(a[0].1.contains("[REDACTED:ssn]"), "redaction marker present");
            }
            o => panic!("expected Redacted, got {o:?}"),
        }
    }
    #[test]
    fn allow_clean_passes_through() {
        let clean = vec![("body".into(), "refill standing order bed 4".into())];
        assert_eq!(
            EgressGuard::evaluate("r", "a", "res", &clean, &store(), &FixedEngine(Effect::Allow)),
            EgressOutcome::Allow
        );
    }
    #[test]
    fn escalate_routes_to_human() {
        assert_eq!(
            EgressGuard::evaluate("r", "a", "res", &leaking(), &store(), &FixedEngine(Effect::Escalate)),
            EgressOutcome::Escalate
        );
    }
}
