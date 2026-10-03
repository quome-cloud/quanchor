//! The regulatory passport artifact: a versioned, ed25519-signed declaration of
//! what an agent was cleared to be — its metadata, the frozen per-jurisdiction
//! classifications, its PCCP, the fingerprint of its cleared configuration, an
//! expiry, and the authorized autonomy envelope. Signing follows the same pattern
//! as the 008 signed registry (`SigningKey::sign` over canonical JSON; production
//! key released via attestation). Loading **fails closed** on a bad/missing
//! signature, expiry, or unparsable body.
//!
//! `Passport::autonomy_envelope()` returns the `monitor::AutonomyEnvelope` that
//! paper 011 modelled locally — closing the 011→012 loop: a deployed drift monitor
//! reads its authorized envelope straight off the passport.

use super::{AgentMetadata, Classification, Jurisdiction};
use crate::monitor::autonomy::AutonomyEnvelope;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// The authorized-autonomy envelope, frozen into the passport. Mirrors the shape of
/// [`crate::monitor::autonomy::AutonomyEnvelope`] (which 011 modelled locally) and
/// is convertible to it via [`Passport::autonomy_envelope`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeSpec {
    /// Highest autonomy tier (0..=3) the agent may actuate autonomously.
    pub max_autonomous_risk_tier: u8,
    /// Maximum fraction of recent actions taken autonomously.
    pub max_autonomous_fraction: f64,
    /// Sliding-window size for the fraction KPI.
    pub window: usize,
}

/// The passport body (the bytes that get signed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Passport {
    /// Binds to the 008 registry: the gate admits only this agent_id.
    pub agent_id: String,
    /// Passport version — stamped into every 003 audit entry (C2.6).
    pub version: String,
    /// The classification rule-set version this passport was cleared under.
    pub ruleset_version: String,
    pub metadata: AgentMetadata,
    /// Per-jurisdiction classifications, frozen at clearance time.
    pub classifications: Vec<(Jurisdiction, Classification)>,
    pub pccp: super::pccp::Pccp,
    /// Fingerprint of the cleared configuration; the gate diffs the live config
    /// against this.
    pub deployed_fingerprint: super::fingerprint::Fingerprint,
    /// Expiry, unix seconds. The gate rejects an expired passport.
    pub not_after: i64,
    pub autonomy_envelope: EnvelopeSpec,
}

impl Passport {
    /// The authorized autonomy envelope as the 011 monitor's type (closes the loop).
    pub fn autonomy_envelope(&self) -> AutonomyEnvelope {
        AutonomyEnvelope {
            max_autonomous_risk_tier: self.autonomy_envelope.max_autonomous_risk_tier,
            max_autonomous_fraction: self.autonomy_envelope.max_autonomous_fraction,
            window: self.autonomy_envelope.window,
        }
    }

    /// Canonical bytes signed over: serde_json of the passport body.
    fn canonical(&self) -> crate::Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Sign this passport with the issuer key (attestation-released in production).
    pub fn sign(&self, issuer: &SigningKey) -> crate::Result<SignedPassport> {
        let bytes = self.canonical()?;
        let sig: Signature = issuer.sign(&bytes);
        Ok(SignedPassport {
            passport_json: String::from_utf8(bytes).expect("serde_json emits utf8"),
            signature_hex: hex::encode(sig.to_bytes()),
            issuer_pubkey_hex: hex::encode(issuer.verifying_key().to_bytes()),
        })
    }
}

/// On-disk passport: the signed body plus the issuer signature and public key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedPassport {
    pub passport_json: String,
    pub signature_hex: String,
    pub issuer_pubkey_hex: String,
}

impl SignedPassport {
    fn issuer_key(&self) -> crate::Result<VerifyingKey> {
        let bytes = hex::decode(&self.issuer_pubkey_hex)
            .map_err(|e| crate::error::Error::Config(format!("bad issuer key: {e}")))?;
        let arr: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| crate::error::Error::Config("issuer key must be 32 bytes".into()))?;
        VerifyingKey::from_bytes(&arr)
            .map_err(|e| crate::error::Error::Config(format!("bad issuer key: {e}")))
    }

    /// Verify the signature only (no expiry check); returns the parsed passport.
    pub fn verify(&self) -> crate::Result<Passport> {
        let issuer = self.issuer_key()?;
        let sig_bytes = hex::decode(&self.signature_hex)
            .map_err(|e| crate::error::Error::Config(format!("bad signature hex: {e}")))?;
        let sig = Signature::from_slice(&sig_bytes)
            .map_err(|e| crate::error::Error::Config(format!("bad signature: {e}")))?;
        issuer
            .verify(self.passport_json.as_bytes(), &sig)
            .map_err(|_| crate::error::Error::Config("passport signature invalid".into()))?;
        let passport: Passport = serde_json::from_str(&self.passport_json)?;
        Ok(passport)
    }

    /// Verify signature *and* expiry against `now` (unix seconds). Fail-closed.
    pub fn verify_at(&self, now: i64) -> crate::Result<Passport> {
        let p = self.verify()?;
        if now > p.not_after {
            return Err(crate::error::Error::Config(format!(
                "passport expired at {} (now {now})",
                p.not_after
            )));
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::fingerprint::Fingerprint;
    use crate::lifecycle::pccp::Pccp;
    use crate::lifecycle::{
        AgentMetadata, AutonomyLevel, ClinicalTask, OutputType, Severity,
    };

    fn passport(not_after: i64) -> Passport {
        Passport {
            agent_id: "agent-1".into(),
            version: "1.0.0".into(),
            ruleset_version: "2026.06".into(),
            metadata: AgentMetadata {
                intended_use: "triage".into(),
                clinical_task: ClinicalTask::Drive,
                output_type: OutputType::Recommendation,
                autonomy: AutonomyLevel::Advisory,
                condition_severity: Severity::Serious,
                patient_facing: false,
                tools: vec!["lookup".into()],
            },
            classifications: vec![],
            pccp: Pccp { allowed: vec![] },
            deployed_fingerprint: Fingerprint::of(b"w", b"p", b"t", b"d"),
            not_after,
            autonomy_envelope: EnvelopeSpec {
                max_autonomous_risk_tier: 2,
                max_autonomous_fraction: 0.3,
                window: 100,
            },
        }
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    #[test]
    fn signed_passport_round_trips() {
        let p = passport(10_000);
        let signed = p.sign(&key()).unwrap();
        let back = signed.verify().unwrap();
        assert_eq!(back, p);
        assert!(signed.verify_at(9_999).is_ok());
    }

    #[test]
    fn tampered_passport_rejected() {
        let mut signed = passport(10_000).sign(&key()).unwrap();
        signed.passport_json = signed.passport_json.replace("agent-1", "agent-evil");
        assert!(signed.verify().is_err());
    }

    #[test]
    fn expired_passport_rejected() {
        let signed = passport(100).sign(&key()).unwrap();
        assert!(signed.verify().is_ok()); // signature fine
        assert!(signed.verify_at(101).is_err()); // but expired
    }

    #[test]
    fn wrong_key_rejected() {
        let mut signed = passport(10_000).sign(&key()).unwrap();
        let other = SigningKey::from_bytes(&[9u8; 32]);
        signed.issuer_pubkey_hex = hex::encode(other.verifying_key().to_bytes());
        assert!(signed.verify().is_err());
    }

    #[test]
    fn autonomy_envelope_maps_to_monitor_type() {
        let p = passport(10_000);
        let env = p.autonomy_envelope();
        assert_eq!(env.max_autonomous_risk_tier, 2);
        assert_eq!(env.max_autonomous_fraction, 0.3);
        assert_eq!(env.window, 100);
    }
}
