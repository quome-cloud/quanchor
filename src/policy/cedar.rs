//! Cedar backend. Translates a Request into a Cedar authorization query against a
//! parsed PolicySet, returning Allow/Deny with the determining policy ids as the
//! matched rule / reasons. Cedar additionally offers schema-based policy
//! validation (used by the `qfire policy coverage`/validate path).
//!
//! **API note (cedar-policy 4.11):** `Request::new()` takes non-optional `EntityUid`
//! for all three components (principal/action/resource). `Context::from_json_value`
//! takes `Option<(&Schema, &EntityUid)>` — we pass `None` (schema-free mode).

use super::{Effect, PolicyDecision, PolicyEngine, Request};
use cedar_policy::{Authorizer, Context, Entities, EntityUid, PolicySet, Request as CedarRequest};

pub struct CedarEngine {
    policies: PolicySet,
    authorizer: Authorizer,
}

impl CedarEngine {
    /// Parse a Cedar policy source string into an engine. Fail-closed: returns
    /// an `Err` on parse failure so the caller can deny rather than panic.
    pub fn from_src(src: &str) -> crate::Result<Self> {
        let policies: PolicySet = src
            .parse()
            .map_err(|e| crate::error::Error::Config(format!("cedar policy parse: {e}")))?;
        Ok(CedarEngine {
            policies,
            authorizer: Authorizer::new(),
        })
    }

    /// Parse a `Type::"id"` entity UID from type-name and id strings.
    fn uid(kind: &str, id: &str) -> crate::Result<EntityUid> {
        format!("{kind}::\"{id}\"")
            .parse()
            .map_err(|e| crate::error::Error::Config(format!("cedar uid {kind}::{id}: {e}")))
    }
}

impl PolicyEngine for CedarEngine {
    fn decide(&self, req: &Request) -> PolicyDecision {
        // Build principal / action / resource UIDs and a Context from args+attrs.
        // Fail-closed: any construction/parse error → deny.
        let build = || -> crate::Result<PolicyDecision> {
            // API change vs. plan sketch: Request::new() in 4.11 takes EntityUid
            // (not Option<EntityUid>) for all three components.
            let principal = Self::uid("Role", &req.principal)?;
            let action = Self::uid("Action", &req.action)?;
            let resource = Self::uid("Resource", &req.resource)?;

            // Build context JSON as a flat record containing `args` and `attrs`.
            let ctx_json = serde_json::json!({ "args": req.args, "attrs": req.attrs });
            // API change: Context::from_json_value takes Option<(&Schema, &EntityUid)>
            // (schema + action uid pair) or None for schema-free parsing.
            let context = Context::from_json_value(ctx_json, None)
                .map_err(|e| crate::error::Error::Config(format!("cedar ctx: {e}")))?;

            // cedar-policy 4.11: Request::new(principal, action, resource, context, schema?)
            let cedar_req = CedarRequest::new(principal, action, resource, context, None)
                .map_err(|e| crate::error::Error::Config(format!("cedar req: {e}")))?;

            let response =
                self.authorizer
                    .is_authorized(&cedar_req, &self.policies, &Entities::empty());

            let effect = match response.decision() {
                cedar_policy::Decision::Allow => Effect::Allow,
                cedar_policy::Decision::Deny => Effect::Deny,
            };

            // Collect determining policy ids from diagnostics.
            let matched: Vec<String> = response
                .diagnostics()
                .reason()
                .map(|p| p.to_string())
                .collect();

            Ok(PolicyDecision {
                effect,
                matched_rule: matched.first().cloned(),
                reasons: matched,
            })
        };

        build().unwrap_or_else(|e| PolicyDecision::deny(format!("cedar error: {e}")))
    }

    fn name(&self) -> &'static str {
        "cedar"
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
            attrs: json!({"active_encounter": true}),
        }
    }

    #[test]
    fn permit_allows_forbid_denies() {
        // A minimal Cedar policy: permit prescribers to order_medication; deny otherwise.
        // Uses Role/Action/Resource entity types (matches the uid() helper above).
        let src = r#"
            permit(
                principal == Role::"prescriber",
                action == Action::"order_medication",
                resource
            );
        "#;
        let eng = CedarEngine::from_src(src).unwrap();
        // A permit rule matches → Allow
        assert_eq!(
            eng.decide(&req("prescriber", "order_medication")).effect,
            Effect::Allow
        );
        // No permit rule matches nurse → Deny (Cedar is deny-by-default)
        assert_eq!(
            eng.decide(&req("nurse", "order_medication")).effect,
            Effect::Deny
        );
    }
}
