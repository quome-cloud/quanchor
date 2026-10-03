//! The signed agent catalog (HAARF C5). A registry file lists every admissible
//! agent (id, role, ed25519 verifying key, capability ceiling, issuer) and is
//! signed by the issuer key. Loading verifies the signature before trusting any
//! entry; unknown agents and tampered files are rejected (fail-closed).

use crate::identity::{AgentId, Capability};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registration {
    pub agent_id: AgentId,
    pub role: String,
    /// ed25519 verifying key, hex-encoded (32 bytes -> 64 hex chars).
    pub pubkey_hex: String,
    /// Capability ceiling: tokens from this agent may only attenuate below this.
    pub max_caps: Vec<Capability>,
    pub issuer: String,
}

impl Registration {
    pub fn verifying_key(&self) -> crate::Result<VerifyingKey> {
        let bytes = hex::decode(&self.pubkey_hex)
            .map_err(|e| crate::error::Error::Config(format!("bad pubkey hex: {e}")))?;
        let arr: [u8; 32] = bytes.as_slice().try_into()
            .map_err(|_| crate::error::Error::Config("pubkey must be 32 bytes".into()))?;
        VerifyingKey::from_bytes(&arr)
            .map_err(|e| crate::error::Error::Config(format!("bad pubkey: {e}")))
    }
}

/// On-disk file shape: the entries plus an issuer signature over their canonical JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryFile {
    pub entries: Vec<Registration>,
    /// hex ed25519 signature by `issuer_pubkey_hex` over canonical_json(entries).
    pub signature_hex: String,
    pub issuer_pubkey_hex: String,
}

/// A verified, queryable registry.
pub struct Registry {
    by_id: HashMap<AgentId, Registration>,
}

/// Canonical bytes signed over: serde_json of the entries vec.
fn canonical(entries: &[Registration]) -> crate::Result<Vec<u8>> {
    Ok(serde_json::to_vec(entries)?)
}

impl Registry {
    /// Parse + verify a registry file. Fails closed on a bad/missing signature.
    pub fn load_verified(file: &RegistryFile) -> crate::Result<Self> {
        let issuer_bytes = hex::decode(&file.issuer_pubkey_hex)
            .map_err(|e| crate::error::Error::Config(format!("bad issuer key: {e}")))?;
        let issuer_arr: [u8; 32] = issuer_bytes.as_slice().try_into()
            .map_err(|_| crate::error::Error::Config("issuer key must be 32 bytes".into()))?;
        let issuer = VerifyingKey::from_bytes(&issuer_arr)
            .map_err(|e| crate::error::Error::Config(format!("bad issuer key: {e}")))?;
        let sig_bytes = hex::decode(&file.signature_hex)
            .map_err(|e| crate::error::Error::Config(format!("bad signature hex: {e}")))?;
        let sig = Signature::from_slice(&sig_bytes)
            .map_err(|e| crate::error::Error::Config(format!("bad signature: {e}")))?;
        issuer.verify(&canonical(&file.entries)?, &sig)
            .map_err(|_| crate::error::Error::Config("registry signature invalid".into()))?;
        let by_id = file.entries.iter().cloned()
            .map(|r| (r.agent_id.clone(), r)).collect();
        Ok(Registry { by_id })
    }

    pub fn get(&self, id: &AgentId) -> Option<&Registration> {
        self.by_id.get(id)
    }
}

/// Test helper: sign a set of entries into a RegistryFile with a fresh issuer key.
pub fn sign_registry(entries: Vec<Registration>, issuer: &SigningKey) -> crate::Result<RegistryFile> {
    let sig: Signature = issuer.sign(&canonical(&entries)?);
    Ok(RegistryFile {
        entries,
        signature_hex: hex::encode(sig.to_bytes()),
        issuer_pubkey_hex: hex::encode(issuer.verifying_key().to_bytes()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn agent(id: &str, key: &SigningKey, caps: &[&str]) -> Registration {
        Registration {
            agent_id: AgentId(id.into()),
            role: "triage".into(),
            pubkey_hex: hex::encode(key.verifying_key().to_bytes()),
            max_caps: caps.iter().map(|c| Capability(c.to_string())).collect(),
            issuer: "test-issuer".into(),
        }
    }

    fn fixed_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    #[test]
    fn loads_and_finds_a_signed_agent() {
        let issuer = fixed_key(1);
        let a = fixed_key(2);
        let file = sign_registry(vec![agent("a1", &a, &["read_phi"])], &issuer).unwrap();
        let reg = Registry::load_verified(&file).unwrap();
        assert_eq!(reg.get(&AgentId("a1".into())).unwrap().role, "triage");
    }

    #[test]
    fn unknown_agent_is_none() {
        let issuer = fixed_key(1);
        let a = fixed_key(2);
        let file = sign_registry(vec![agent("a1", &a, &[])], &issuer).unwrap();
        let reg = Registry::load_verified(&file).unwrap();
        assert!(reg.get(&AgentId("ghost".into())).is_none());
    }

    #[test]
    fn tampered_entry_breaks_signature() {
        let issuer = fixed_key(1);
        let a = fixed_key(2);
        let mut file = sign_registry(vec![agent("a1", &a, &["read_phi"])], &issuer).unwrap();
        file.entries[0].max_caps.push(Capability("order_medication".into()));
        assert!(Registry::load_verified(&file).is_err());
    }
}
