//! Attenuable agent capability tokens (biscuit-auth). The issuer mints a token
//! granting an agent a capability set, bound to its `AgentId` (audience). A
//! handoff appends a restriction block — attenuation is monotonic: caps can only
//! shrink. The gateway authorizes the *effective* (post-attenuation) caps against
//! the registered ceiling, defeating impersonation (T1) and confused-deputy (T3).

use crate::identity::{AgentId, Capability};

#[cfg(test)]
mod smoke {
    use biscuit_auth::{Biscuit, KeyPair};
    use biscuit_auth::macros::biscuit;

    #[test]
    fn biscuit_roundtrips() {
        let root = KeyPair::new();
        let token = biscuit!(r#"cap("read_phi"); audience("a1");"#).build(&root).unwrap();
        let b64 = token.to_base64().unwrap();
        let parsed = Biscuit::from_base64(&b64, root.public()).unwrap();
        assert_eq!(parsed.block_count(), 1);
    }
}

use biscuit_auth::{Biscuit, KeyPair, PublicKey, AuthorizerBuilder, BlockBuilder};

/// Mint a token granting `caps` to `agent`, signed by the issuer root keypair.
pub fn mint(root: &KeyPair, agent: &AgentId, caps: &[Capability]) -> crate::Result<String> {
    let mut builder = Biscuit::builder();
    builder = builder
        .fact(format!(r#"audience("{}")"#, agent.0).as_str())
        .map_err(|e| crate::error::Error::Config(format!("biscuit fact: {e}")))?;
    for c in caps {
        builder = builder
            .fact(format!(r#"cap("{}")"#, c.0).as_str())
            .map_err(|e| crate::error::Error::Config(format!("biscuit fact: {e}")))?;
    }
    let token = builder
        .build(root)
        .map_err(|e| crate::error::Error::Config(format!("biscuit build: {e}")))?;
    token
        .to_base64()
        .map_err(|e| crate::error::Error::Config(format!("biscuit b64: {e}")))
}

/// Attenuate a token by restricting its caps to `keep` (a subset of current caps).
pub fn attenuate(token_b64: &str, root: PublicKey, keep: &[Capability]) -> crate::Result<String> {
    let token = Biscuit::from_base64(token_b64, root)
        .map_err(|e| crate::error::Error::Config(format!("biscuit parse: {e}")))?;
    let allowed = keep
        .iter()
        .map(|c| format!(r#""{}""#, c.0))
        .collect::<Vec<_>>()
        .join(", ");
    // "check all" = for every cap fact, it must be within the keep set.
    // This is the monotonic-attenuation invariant: narrowing blocks can only shrink caps.
    let restriction = format!(r#"check all cap($c), [{allowed}].contains($c)"#);
    let bb = BlockBuilder::new()
        .check(restriction.as_str())
        .map_err(|e| crate::error::Error::Config(format!("biscuit check: {e}")))?;
    let attenuated = token
        .append(bb)
        .map_err(|e| crate::error::Error::Config(format!("biscuit append: {e}")))?;
    attenuated
        .to_base64()
        .map_err(|e| crate::error::Error::Config(format!("biscuit b64: {e}")))
}

/// Authorize: verify under `root`, confirm holder==audience, action is a granted
/// cap, and effective caps are within `ceiling`. Returns effective caps.
pub fn authorize(
    token_b64: &str,
    root: PublicKey,
    holder: &AgentId,
    action: &Capability,
    ceiling: &[Capability],
) -> crate::Result<Vec<Capability>> {
    let token = Biscuit::from_base64(token_b64, root)
        .map_err(|_| crate::error::Error::Config("token: invalid signature".into()))?;
    let mut authz = AuthorizerBuilder::new()
        .fact(format!(r#"request_holder("{}")"#, holder.0).as_str())
        .map_err(|e| crate::error::Error::Config(format!("fact: {e}")))?
        // AuthorizerBuilder::check() parses a single check (no trailing semicolon)
        .check(r#"check if audience($a), request_holder($a)"#)
        .map_err(|e| crate::error::Error::Config(format!("check: {e}")))?
        .fact(format!(r#"requested("{}")"#, action.0).as_str())
        .map_err(|e| crate::error::Error::Config(format!("fact: {e}")))?
        .check(r#"check if requested($r), cap($r)"#)
        .map_err(|e| crate::error::Error::Config(format!("check: {e}")))?
        .policy("allow if true")
        .map_err(|e| crate::error::Error::Config(format!("policy: {e}")))?
        .build(&token)
        .map_err(|e| crate::error::Error::Config(format!("authorizer: {e}")))?;
    authz
        .authorize()
        .map_err(|_| crate::error::Error::Config("token: authorization denied".into()))?;
    let rows: Vec<(String,)> = authz
        .query("data($c) <- cap($c)")
        .map_err(|e| crate::error::Error::Config(format!("query: {e}")))?;
    let ceil: std::collections::HashSet<_> = ceiling.iter().cloned().collect();
    let eff = rows
        .into_iter()
        .map(|(c,)| Capability(c))
        .filter(|c| ceil.contains(c))
        .collect();
    Ok(eff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use biscuit_auth::KeyPair;

    fn caps(v: &[&str]) -> Vec<Capability> {
        v.iter().map(|s| Capability(s.to_string())).collect()
    }

    #[test]
    fn granted_action_authorizes() {
        let root = KeyPair::new();
        let a = AgentId("a1".into());
        let t = mint(&root, &a, &caps(&["read_phi", "document"])).unwrap();
        let eff = authorize(
            &t,
            root.public(),
            &a,
            &Capability("read_phi".into()),
            &caps(&["read_phi", "document"]),
        )
        .unwrap();
        assert!(eff.contains(&Capability("read_phi".into())));
    }

    #[test]
    fn ungranted_action_is_denied() {
        let root = KeyPair::new();
        let a = AgentId("a1".into());
        let t = mint(&root, &a, &caps(&["read_phi"])).unwrap();
        let r = authorize(
            &t,
            root.public(),
            &a,
            &Capability("order_medication".into()),
            &caps(&["read_phi"]),
        );
        assert!(r.is_err(), "ungranted action must be denied (T1/T3)");
    }

    #[test]
    fn attenuation_cannot_re_add_caps() {
        let root = KeyPair::new();
        let a = AgentId("a1".into());
        let t = mint(&root, &a, &caps(&["read_phi", "order_medication"])).unwrap();
        let narrowed = attenuate(&t, root.public(), &caps(&["read_phi"])).unwrap();
        let r = authorize(
            &narrowed,
            root.public(),
            &a,
            &Capability("order_medication".into()),
            &caps(&["read_phi", "order_medication"]),
        );
        assert!(r.is_err(), "attenuation must be monotonic (T3 confused-deputy)");
    }

    #[test]
    fn stolen_token_rejected_for_wrong_holder() {
        let root = KeyPair::new();
        let a = AgentId("a1".into());
        let t = mint(&root, &a, &caps(&["read_phi"])).unwrap();
        let r = authorize(
            &t,
            root.public(),
            &AgentId("a2".into()),
            &Capability("read_phi".into()),
            &caps(&["read_phi"]),
        );
        assert!(r.is_err(), "holder-of-key binding must reject a stolen token (T1)");
    }
}
