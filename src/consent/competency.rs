//! consent::competency — G3 credentialed-operator gate. Certain action classes
//! require a credentialed human in the loop (C4.5, C5.6). The gate honours an
//! [`OperatorAttestation`] only when its ed25519 signature verifies against the
//! operator's registered key; an unsigned, mis-signed, missing, or
//! under-credentialed operator escalates the action to human review.
//!
//! Signing reuses the 008 ed25519 plumbing: the signed message is the canonical
//! `operator_id|c1,c2,...` with credentials sorted (the `BTreeSet` ordering).

use super::{AgentAction, ConsentControl, ConsentControlFn, ConsentDecision, ConsentEffect, OperatorAttestation};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use std::collections::{BTreeSet, HashMap};

/// The canonical bytes signed by an operator attestation.
pub fn signed_bytes(operator_id: &str, credentials: &BTreeSet<String>) -> Vec<u8> {
    let creds = credentials.iter().cloned().collect::<Vec<_>>().join(",");
    format!("{operator_id}|{creds}").into_bytes()
}

/// Build a signed attestation (used by the dataset generator and tests).
pub fn sign_attestation(
    key: &SigningKey,
    operator_id: &str,
    credentials: BTreeSet<String>,
) -> OperatorAttestation {
    let sig: Signature = key.sign(&signed_bytes(operator_id, &credentials));
    OperatorAttestation {
        operator_id: operator_id.to_string(),
        credentials,
        signature: hex::encode(sig.to_bytes()),
    }
}

/// The G3 gate: which credential each action class requires, plus the registry of
/// operator verifying keys used to authenticate attestations.
#[derive(Debug, Clone, Default)]
pub struct CompetencyGate {
    /// action class → required credential (absent ⇒ no credential required).
    required: HashMap<String, String>,
    /// operator_id → hex ed25519 verifying key.
    keys: HashMap<String, String>,
}

impl CompetencyGate {
    pub fn new() -> Self {
        CompetencyGate::default()
    }

    /// Require `credential` for `action` class.
    pub fn require(&mut self, action: &str, credential: &str) -> &mut Self {
        self.required.insert(action.to_string(), credential.to_string());
        self
    }

    /// Register an operator's verifying key (hex).
    pub fn register_key(&mut self, operator_id: &str, pubkey_hex: &str) -> &mut Self {
        self.keys.insert(operator_id.to_string(), pubkey_hex.to_string());
        self
    }

    fn verify(&self, att: &OperatorAttestation) -> bool {
        let key_hex = match self.keys.get(&att.operator_id) {
            Some(k) => k,
            None => return false,
        };
        let key = match hex::decode(key_hex)
            .ok()
            .and_then(|b| <[u8; 32]>::try_from(b).ok())
            .and_then(|arr| VerifyingKey::from_bytes(&arr).ok())
        {
            Some(k) => k,
            None => return false,
        };
        let sig = match hex::decode(&att.signature)
            .ok()
            .and_then(|b| Signature::from_slice(&b).ok())
        {
            Some(s) => s,
            None => return false,
        };
        key.verify(&signed_bytes(&att.operator_id, &att.credentials), &sig)
            .is_ok()
    }
}

impl ConsentControlFn for CompetencyGate {
    fn decide(&self, action: &AgentAction) -> ConsentDecision {
        // No credential required for this action class → not our concern.
        let required = match self.required.get(&action.action) {
            Some(c) => c,
            None => return ConsentDecision::allow(),
        };

        let att = match &action.operator {
            Some(a) => a,
            None => {
                return ConsentDecision::stop(
                    ConsentEffect::Escalate,
                    ConsentControl::Competency,
                    format!("action `{}` requires credential `{required}` but no operator attestation present", action.action),
                )
            }
        };

        if !self.verify(att) {
            return ConsentDecision::stop(
                ConsentEffect::Escalate,
                ConsentControl::Competency,
                format!("operator `{}` attestation failed signature verification", att.operator_id),
            );
        }
        if !att.credentials.contains(required) {
            return ConsentDecision::stop(
                ConsentEffect::Escalate,
                ConsentControl::Competency,
                format!("operator `{}` lacks required credential `{required}`", att.operator_id),
            );
        }
        ConsentDecision::allow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn gate_with(op: &str, key: &SigningKey) -> CompetencyGate {
        let mut g = CompetencyGate::new();
        g.require("order_controlled_substance", "DEA")
            .register_key(op, &hex::encode(key.verifying_key().to_bytes()));
        g
    }

    fn action_by(op: Option<OperatorAttestation>) -> AgentAction {
        let mut a = AgentAction::new("c", "p1", "order_controlled_substance", "treatment");
        a.operator = op;
        a
    }

    #[test]
    fn uncredentialed_action_class_is_allowed() {
        let g = CompetencyGate::new();
        let a = AgentAction::new("c", "p1", "send_reminder", "contact");
        assert_eq!(g.decide(&a).effect, ConsentEffect::Allow);
    }

    #[test]
    fn valid_attestation_with_credential_allows() {
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let g = gate_with("dr-jones", &key);
        let att = sign_attestation(&key, "dr-jones", set(&["DEA", "MD"]));
        assert_eq!(g.decide(&action_by(Some(att))).effect, ConsentEffect::Allow);
    }

    #[test]
    fn missing_attestation_escalates() {
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let g = gate_with("dr-jones", &key);
        assert_eq!(g.decide(&action_by(None)).effect, ConsentEffect::Escalate);
    }

    #[test]
    fn missing_credential_escalates() {
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let g = gate_with("dr-jones", &key);
        let att = sign_attestation(&key, "dr-jones", set(&["MD"])); // no DEA
        assert_eq!(g.decide(&action_by(Some(att))).effect, ConsentEffect::Escalate);
    }

    #[test]
    fn forged_signature_escalates() {
        let real = SigningKey::from_bytes(&[7u8; 32]);
        let attacker = SigningKey::from_bytes(&[9u8; 32]);
        let g = gate_with("dr-jones", &real);
        // Attacker signs a DEA attestation in dr-jones's name with the wrong key.
        let forged = sign_attestation(&attacker, "dr-jones", set(&["DEA"]));
        let d = g.decide(&action_by(Some(forged)));
        assert_eq!(d.effect, ConsentEffect::Escalate);
        assert!(d.reasons[0].contains("signature verification"));
    }

    #[test]
    fn tampered_credentials_break_signature() {
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let g = gate_with("dr-jones", &key);
        // Sign with only MD, then tamper the set to add DEA after signing.
        let mut att = sign_attestation(&key, "dr-jones", set(&["MD"]));
        att.credentials.insert("DEA".to_string());
        assert_eq!(g.decide(&action_by(Some(att))).effect, ConsentEffect::Escalate);
    }
}
