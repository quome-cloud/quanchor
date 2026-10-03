//! Paper 008 — agent identity for multi-agent systems. This layer runs *before*
//! `policy` (005): it replaces the unauthenticated header principal with a
//! cryptographically-proven `AgentId` and an attenuated capability set, verifies
//! per-message provenance, charges a fleet-global capability accountant, and
//! enforces quorum for approval-gated actions. Fail-closed; decisions are stamped
//! into the 003 audit chain. Defeats threats T1–T5 (see the feature spec).

pub mod accountant;
pub mod envelope;
pub mod quorum;
pub mod registry;
pub mod token;

use serde::{Deserialize, Serialize};

/// A stable agent handle, keyed in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

/// A capability is the right to perform one action-class (e.g. "order_medication").
/// An agent's registered ceiling is a set of these; tokens may only attenuate
/// (subset) it, never escalate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Capability(pub String);

// ─── Task 6: IdentityLayer::check ───────────────────────────────────────────

use crate::identity::accountant::Accountant;
use crate::identity::registry::Registry;
use biscuit_auth::PublicKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityEffect { Allow, Deny }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult { pub name: String, pub ok: bool, pub reason: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityDecision {
    pub effect: IdentityEffect,
    pub checks: Vec<CheckResult>,
    pub proven_principal: Option<AgentId>,
    pub attenuated_caps: Vec<Capability>,
}

/// Approval requirement for an approval-gated action: k distinct, registered approvers.
#[derive(Debug, Clone)]
pub struct Approval { pub approvers: Vec<AgentId>, pub k: usize }

/// One inter-agent hop presented to the gateway.
pub struct Handoff {
    pub from: AgentId,
    pub token_b64: String,
    pub envelope: crate::identity::envelope::SignedEnvelope,
    pub prior_provenance: Vec<String>,
    pub this_hop: String,
    pub action: Capability,
    pub approval: Option<Approval>,
}

/// The composed identity layer.
pub struct IdentityLayer<'a> {
    pub registry: &'a Registry,
    pub root: PublicKey,
    pub accountant: &'a mut Accountant,
}

impl<'a> IdentityLayer<'a> {
    /// Fail-closed: the first failed check denies, but all evaluated checks are
    /// recorded for the audit trace. On Allow, returns the proven principal +
    /// effective caps for `policy`.
    pub fn check(&mut self, h: &Handoff) -> IdentityDecision {
        let mut checks = Vec::new();

        // 1. Sender must be a registered agent (T1 base).
        let reg = match self.registry.get(&h.from) {
            Some(r) => r.clone(),
            None => {
                checks.push(CheckResult { name: "registered".into(), ok: false,
                    reason: Some(format!("unknown agent {}", h.from.0)) });
                return deny(checks);
            }
        };
        checks.push(CheckResult { name: "registered".into(), ok: true, reason: None });

        // 2. Envelope signature under the sender's registered key (T2).
        let vk = match reg.verifying_key() {
            Ok(k) => k,
            Err(e) => { checks.push(CheckResult{name:"sender_key".into(),ok:false,reason:Some(e.to_string())}); return deny(checks); }
        };
        match crate::identity::envelope::verify_signature(&h.envelope, &vk) {
            Ok(()) => checks.push(CheckResult { name: "envelope_sig".into(), ok: true, reason: None }),
            Err(e) => { checks.push(CheckResult{name:"envelope_sig".into(),ok:false,reason:Some(e.to_string())}); return deny(checks); }
        }

        // 3. Provenance chain extends prior by exactly this hop (T5).
        match crate::identity::envelope::verify_provenance(&h.envelope, &h.prior_provenance, &h.this_hop) {
            Ok(()) => checks.push(CheckResult { name: "provenance".into(), ok: true, reason: None }),
            Err(e) => { checks.push(CheckResult{name:"provenance".into(),ok:false,reason:Some(e.to_string())}); return deny(checks); }
        }

        // 4. Capability token: holder-bound, attenuation-respecting, action granted, within ceiling (T1/T3).
        //    Two conditions must BOTH hold:
        //    a) token::authorize returns Ok (token valid, holder-bound, action is a granted cap), AND
        //    b) the requested action is within the agent's registered ceiling (registry root-of-trust).
        //    Condition (b) closes the T1 attack where an adversary mints their own token granting a
        //    cap above the registry ceiling — token::authorize would accept it (the cap is in the token),
        //    but the registry ceiling is the authoritative bound.
        if !reg.max_caps.contains(&h.action) {
            checks.push(CheckResult {
                name: "capability".into(),
                ok: false,
                reason: Some(format!("action '{}' exceeds registered ceiling", h.action.0)),
            });
            return deny(checks);
        }
        let eff = match crate::identity::token::authorize(&h.token_b64, self.root, &h.from, &h.action, &reg.max_caps) {
            Ok(eff) => eff,
            Err(e) => { checks.push(CheckResult{name:"capability".into(),ok:false,reason:Some(e.to_string())}); return deny(checks); }
        };
        checks.push(CheckResult { name: "capability".into(), ok: true, reason: None });

        // 5. Fleet capability budget (T4).
        match self.accountant.charge(&h.action) {
            Ok(()) => checks.push(CheckResult { name: "fleet_budget".into(), ok: true, reason: None }),
            Err(e) => { checks.push(CheckResult{name:"fleet_budget".into(),ok:false,reason:Some(e.to_string())}); return deny(checks); }
        }

        // 6. Approval-gated actions require k distinct attested identities (T4 manufactured quorum).
        if let Some(ap) = &h.approval {
            match crate::identity::quorum::check_quorum(&ap.approvers, ap.k, |a| self.registry.get(a).is_some()) {
                Ok(()) => checks.push(CheckResult { name: "quorum".into(), ok: true, reason: None }),
                Err(e) => { checks.push(CheckResult{name:"quorum".into(),ok:false,reason:Some(e.to_string())}); return deny(checks); }
            }
        }

        IdentityDecision { effect: IdentityEffect::Allow, checks,
            proven_principal: Some(h.from.clone()), attenuated_caps: eff }
    }
}

fn deny(checks: Vec<CheckResult>) -> IdentityDecision {
    IdentityDecision { effect: IdentityEffect::Deny, checks, proven_principal: None, attenuated_caps: vec![] }
}

#[cfg(test)]
mod check_tests {
    use super::*;
    use crate::identity::registry::{sign_registry, Registration, Registry};
    use biscuit_auth::KeyPair;
    use ed25519_dalek::SigningKey;
    use std::collections::HashMap;

    fn caps(v: &[&str]) -> Vec<Capability> { v.iter().map(|s| Capability(s.to_string())).collect() }

    struct Fixture { reg: Registry, root: KeyPair, agent_key: SigningKey, token: String }

    fn fixture(agent: &str, ceiling: &[&str], token_caps: &[&str]) -> Fixture {
        let issuer = SigningKey::from_bytes(&[1u8; 32]);
        let agent_key = SigningKey::from_bytes(&[2u8; 32]);
        let entry = Registration {
            agent_id: AgentId(agent.into()), role: "triage".into(),
            pubkey_hex: hex::encode(agent_key.verifying_key().to_bytes()),
            max_caps: caps(ceiling), issuer: "t".into(),
        };
        let file = sign_registry(vec![entry], &issuer).unwrap();
        let reg = Registry::load_verified(&file).unwrap();
        let root = KeyPair::new();
        let token = crate::identity::token::mint(&root, &AgentId(agent.into()), &caps(token_caps)).unwrap();
        Fixture { reg, root, agent_key, token }
    }

    #[test]
    fn legit_handoff_allows() {
        let f = fixture("a1", &["read_phi", "document"], &["read_phi"]);
        let env = crate::identity::envelope::seal(&f.agent_key, &AgentId("a1".into()), "ok", vec!["h0".into()]).unwrap();
        let mut acct = Accountant::new(HashMap::new());
        let mut layer = IdentityLayer { registry: &f.reg, root: f.root.public(), accountant: &mut acct };
        let h = Handoff { from: AgentId("a1".into()), token_b64: f.token, envelope: env,
            prior_provenance: vec![], this_hop: "h0".into(), action: Capability("read_phi".into()),
            approval: None };
        let d = layer.check(&h);
        assert_eq!(d.effect, IdentityEffect::Allow, "{:?}", d.checks);
        assert_eq!(d.proven_principal, Some(AgentId("a1".into())));
    }

    #[test]
    fn unknown_agent_denied() {
        let f = fixture("a1", &["read_phi"], &["read_phi"]);
        let env = crate::identity::envelope::seal(&f.agent_key, &AgentId("ghost".into()), "x", vec!["h0".into()]).unwrap();
        let mut acct = Accountant::new(HashMap::new());
        let mut layer = IdentityLayer { registry: &f.reg, root: f.root.public(), accountant: &mut acct };
        let h = Handoff { from: AgentId("ghost".into()), token_b64: f.token, envelope: env,
            prior_provenance: vec![], this_hop: "h0".into(), action: Capability("read_phi".into()),
            approval: None };
        assert_eq!(layer.check(&h).effect, IdentityEffect::Deny);
    }

    #[test]
    fn t1_over_claim_denied_by_ceiling() {
        // T1 attack: agent "a1" has ceiling ["read_phi"] but mints a token with
        // cap ["order_medication"]. The registry ceiling gate must deny the request
        // even though the token's internal cap check would pass.
        let f = fixture("a1", &["read_phi"], &["order_medication"]);
        let env = crate::identity::envelope::seal(&f.agent_key, &AgentId("a1".into()), "x", vec!["h0".into()]).unwrap();
        let mut acct = Accountant::new(HashMap::new());
        let mut layer = IdentityLayer { registry: &f.reg, root: f.root.public(), accountant: &mut acct };
        let h = Handoff { from: AgentId("a1".into()), token_b64: f.token, envelope: env,
            prior_provenance: vec![], this_hop: "h0".into(), action: Capability("order_medication".into()),
            approval: None };
        let d = layer.check(&h);
        assert_eq!(d.effect, IdentityEffect::Deny, "T1 over-claim must be denied: {:?}", d.checks);
        assert!(d.checks.iter().any(|c| c.name == "capability" && !c.ok),
            "capability check must be the failing check: {:?}", d.checks);
    }

    #[test]
    fn ungranted_action_denied_confused_deputy() {
        let f = fixture("a1", &["read_phi", "order_medication"], &["read_phi"]);
        let env = crate::identity::envelope::seal(&f.agent_key, &AgentId("a1".into()), "x", vec!["h0".into()]).unwrap();
        let mut acct = Accountant::new(HashMap::new());
        let mut layer = IdentityLayer { registry: &f.reg, root: f.root.public(), accountant: &mut acct };
        let h = Handoff { from: AgentId("a1".into()), token_b64: f.token, envelope: env,
            prior_provenance: vec![], this_hop: "h0".into(), action: Capability("order_medication".into()),
            approval: None };
        let d = layer.check(&h);
        assert_eq!(d.effect, IdentityEffect::Deny);
        assert!(d.checks.iter().any(|c| c.name == "capability" && !c.ok));
    }

    /// Build a two-agent registry (a1 and a2) for quorum tests.
    fn two_agent_registry() -> (Registry, KeyPair, SigningKey, SigningKey, String) {
        let issuer = SigningKey::from_bytes(&[1u8; 32]);
        let key_a1 = SigningKey::from_bytes(&[2u8; 32]);
        let key_a2 = SigningKey::from_bytes(&[3u8; 32]);
        let entries = vec![
            Registration {
                agent_id: AgentId("a1".into()), role: "triage".into(),
                pubkey_hex: hex::encode(key_a1.verifying_key().to_bytes()),
                max_caps: caps(&["read_phi"]), issuer: "t".into(),
            },
            Registration {
                agent_id: AgentId("a2".into()), role: "triage".into(),
                pubkey_hex: hex::encode(key_a2.verifying_key().to_bytes()),
                max_caps: caps(&["read_phi"]), issuer: "t".into(),
            },
        ];
        let file = sign_registry(entries, &issuer).unwrap();
        let reg = Registry::load_verified(&file).unwrap();
        let root = KeyPair::new();
        let token = crate::identity::token::mint(&root, &AgentId("a1".into()), &caps(&["read_phi"])).unwrap();
        (reg, root, key_a1, key_a2, token)
    }

    #[test]
    fn quorum_met_allows() {
        let (reg, root, key_a1, _key_a2, token) = two_agent_registry();
        let env = crate::identity::envelope::seal(&key_a1, &AgentId("a1".into()), "ok", vec!["h0".into()]).unwrap();
        let mut acct = Accountant::new(HashMap::new());
        let mut layer = IdentityLayer { registry: &reg, root: root.public(), accountant: &mut acct };
        let h = Handoff {
            from: AgentId("a1".into()), token_b64: token, envelope: env,
            prior_provenance: vec![], this_hop: "h0".into(), action: Capability("read_phi".into()),
            approval: Some(Approval { approvers: vec![AgentId("a1".into()), AgentId("a2".into())], k: 2 }),
        };
        let d = layer.check(&h);
        assert_eq!(d.effect, IdentityEffect::Allow, "quorum met should Allow: {:?}", d.checks);
        assert!(d.checks.iter().any(|c| c.name == "quorum" && c.ok),
            "quorum check should be ok:true: {:?}", d.checks);
    }

    #[test]
    fn sybil_quorum_denied() {
        let (reg, root, key_a1, _key_a2, token) = two_agent_registry();
        let env = crate::identity::envelope::seal(&key_a1, &AgentId("a1".into()), "ok", vec!["h0".into()]).unwrap();
        let mut acct = Accountant::new(HashMap::new());
        let mut layer = IdentityLayer { registry: &reg, root: root.public(), accountant: &mut acct };
        let h = Handoff {
            from: AgentId("a1".into()), token_b64: token, envelope: env,
            prior_provenance: vec![], this_hop: "h0".into(), action: Capability("read_phi".into()),
            // a1 duplicated = Sybil: two copies of same agent, only 1 distinct → quorum not met
            approval: Some(Approval { approvers: vec![AgentId("a1".into()), AgentId("a1".into())], k: 2 }),
        };
        let d = layer.check(&h);
        assert_eq!(d.effect, IdentityEffect::Deny, "Sybil quorum must Deny: {:?}", d.checks);
        assert!(d.checks.iter().any(|c| c.name == "quorum" && !c.ok),
            "quorum check should be ok:false: {:?}", d.checks);
    }
}

// ─── Task 7: Audit stamping helper ──────────────────────────────────────────

/// Serialize a decision for the 003 audit chain (authorization entry kind).
pub fn audit_body(h_from: &AgentId, action: &Capability, d: &IdentityDecision) -> String {
    serde_json::json!({
        "layer": "identity",
        "from": h_from.0,
        "action": action.0,
        "effect": d.effect,
        "checks": d.checks,
        "proven_principal": d.proven_principal,
    }).to_string()
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    #[test]
    fn audit_body_serializes_effect_and_checks() {
        let d = IdentityDecision { effect: IdentityEffect::Deny,
            checks: vec![CheckResult{name:"registered".into(),ok:false,reason:Some("unknown".into())}],
            proven_principal: None, attenuated_caps: vec![] };
        let s = audit_body(&AgentId("a1".into()), &Capability("read_phi".into()), &d);
        assert!(s.contains("\"effect\":\"deny\""));
        assert!(s.contains("\"layer\":\"identity\""));
        assert!(s.contains("registered"));
    }
}

// ─── Task 11: Config surface (request-path wiring seam) ─────────────────────

/// Identity-layer configuration. This is the config surface for the gateway
/// request-path integration, which is deferred (the identity layer runs before
/// `policy` in the gateway path; see the feature spec). In production (the
/// confidential enclave) `registry_path` points at a signed agent catalog
/// released via attestation and the layer fails closed; in dev it is empty
/// (layer disabled). Mirrors `policy::PolicyCfg`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct IdentityCfg {
    /// Path to the signed registry file ("" disables the identity layer).
    pub registry_path: String,
    /// hex ed25519 issuer pubkey for verifying the registry ("" = none).
    pub issuer_pubkey_hex: String,
    /// hex biscuit root public key for verifying capability tokens ("" = none).
    pub root_pubkey_hex: String,
}

#[cfg(test)]
mod cfg_tests {
    use super::*;
    #[test]
    fn cfg_defaults_disabled() {
        let c = IdentityCfg::default();
        assert!(c.registry_path.is_empty());
        assert!(c.issuer_pubkey_hex.is_empty());
        assert!(c.root_pubkey_hex.is_empty());
    }
}
