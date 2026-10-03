//! Rego (OPA) backend via regorus. Loads a Rego module exposing `data.authz.allow`
//! (bool) and optionally `data.authz.effect` ("allow"|"deny"|"escalate") and
//! `data.authz.rule` (string id). Input is { principal, action, resource, args, attrs }.
//!
//! A fresh engine is constructed per decision so calls cannot leak input state.
//! Fail-closed: any engine or evaluation error → PolicyDecision::deny.

use super::{Effect, PolicyDecision, PolicyEngine, Request};

pub struct RegoEngine {
    src: String,
}

impl RegoEngine {
    /// Create a RegoEngine from a Rego policy source string.
    pub fn from_src(src: &str) -> Self {
        RegoEngine {
            src: src.to_string(),
        }
    }

    fn fresh_engine(&self) -> crate::Result<regorus::Engine> {
        let mut e = regorus::Engine::new();
        // add_policy returns anyhow::Result<String> (package name); map to our Error.
        e.add_policy("authz.rego".to_string(), self.src.clone())
            .map_err(|err| crate::error::Error::Config(format!("rego policy load: {err}")))?;
        Ok(e)
    }
}

impl PolicyEngine for RegoEngine {
    fn decide(&self, req: &Request) -> PolicyDecision {
        let build = || -> crate::Result<PolicyDecision> {
            let mut engine = self.fresh_engine()?;

            // Build input as a serde_json Value then convert to regorus::Value via JSON string.
            let input_json = serde_json::json!({
                "principal": req.principal,
                "action": req.action,
                "resource": req.resource,
                "args": req.args,
                "attrs": req.attrs,
            });
            let rval = regorus::Value::from_json_str(&input_json.to_string())
                .map_err(|e| crate::error::Error::Config(format!("rego input parse: {e}")))?;
            engine.set_input(rval);

            // Prefer an explicit `data.authz.effect` rule; fall back to boolean `data.authz.allow`.
            let effect = match eval_string(&mut engine, "data.authz.effect") {
                Some(s) if s == "allow" => Effect::Allow,
                Some(s) if s == "escalate" => Effect::Escalate,
                Some(_) => Effect::Deny,
                None => {
                    if eval_bool(&mut engine, "data.authz.allow") {
                        Effect::Allow
                    } else {
                        Effect::Deny
                    }
                }
            };

            let matched = eval_string(&mut engine, "data.authz.rule");
            Ok(PolicyDecision {
                effect,
                matched_rule: matched,
                reasons: Vec::new(),
            })
        };
        // Fail-closed: any error denies.
        build().unwrap_or_else(|e| PolicyDecision::deny(format!("rego error: {e}")))
    }

    fn name(&self) -> &'static str {
        "rego"
    }
}

/// Evaluate a boolean rule. Returns true only if the result is exactly `Value::Bool(true)`.
/// Any error or non-bool value (including `Value::Undefined`) returns false.
fn eval_bool(engine: &mut regorus::Engine, rule: &str) -> bool {
    matches!(
        engine.eval_rule(rule.to_string()),
        Ok(regorus::Value::Bool(true))
    )
}

/// Evaluate a string rule. Returns `Some(String)` only if the result is `Value::String(_)`.
/// Returns `None` for `Value::Undefined`, errors, or non-string results.
fn eval_string(engine: &mut regorus::Engine, rule: &str) -> Option<String> {
    match engine.eval_rule(rule.to_string()) {
        Ok(regorus::Value::String(s)) => Some(s.as_ref().to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn allow_rule_decides() {
        // Rego v1 syntax (regorus 0.10 defaults to rego_v1 = true).
        let src = r#"
            package authz

            default allow := false

            allow if {
                input.principal == "prescriber"
                input.action == "order_medication"
            }
        "#;
        let eng = RegoEngine::from_src(src);
        assert_eq!(
            eng.decide(&req("prescriber", "order_medication")).effect,
            Effect::Allow
        );
        assert_eq!(
            eng.decide(&req("nurse", "order_medication")).effect,
            Effect::Deny
        );
    }
}
