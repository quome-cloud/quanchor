//! The sealed, signed goal anchor: the sanctioned MIIM (Master Instruction /
//! Intent Manifest) pinned at attestation time, against which live intent drift
//! is measured.
//!
//! # Security note (Plan 3)
//! The seal is `HMAC-SHA256(key, miim)` — length-extension safe. Unlike a plain
//! keyed hash (`SHA256(key || separator || miim)`), HMAC's two-pass construction
//! means a holder of a valid `(miim, seal)` pair cannot forge a seal for an
//! extended `miim` without the key. This matters now that anchors are released
//! from attested / KMS state (Plan 3) and the seal is the real root-of-trust for
//! externally-deserialized anchors.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use subtle::ConstantTimeEq;

/// A goal anchor: the sanctioned MIIM text plus a seal binding it to a secret
/// key known only inside the attested workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalAnchor {
    /// The sanctioned MIIM (master instruction / intent) text.
    pub miim: String,
    /// Hex seal = HMAC-SHA256(key, miim), proving in-enclave sealing.
    pub seal: String,
}

impl GoalAnchor {
    /// Seal a MIIM under a key (called once, inside the attested workload).
    pub fn seal(miim: &str, key: &str) -> Self {
        GoalAnchor { miim: miim.to_string(), seal: seal_hex(key, miim) }
    }

    /// Verify the seal against the key — detects a swapped/forged/tampered anchor.
    /// Uses a constant-time comparison so the result's timing does not leak how
    /// many leading bytes of the seal matched (safe even if exposed over a network).
    pub fn verify(&self, key: &str) -> bool {
        let expected = seal_hex(key, &self.miim);
        let a = expected.as_bytes();
        let b = self.seal.as_bytes();
        a.len() == b.len() && bool::from(a.ct_eq(b))
    }

    /// Lexical drift in `[0,1]` between the anchor MIIM and a live intent string.
    /// 0.0 = identical token set, 1.0 = disjoint. Deterministic, no network.
    pub fn drift(&self, live: &str) -> f64 {
        1.0 - jaccard(&self.miim, live)
    }
}

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Seal = hex(HMAC-SHA256(key, miim)). Unlike a plain keyed hash, HMAC is not
/// length-extension forgeable, so a holder of a valid (miim, seal) pair cannot
/// forge a seal for an extended miim without the key. This matters once anchors
/// are released from attested / KMS state (Plan 3).
fn seal_hex(key: &str, miim: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(key.as_bytes())
        .expect("HMAC accepts keys of any length");
    mac.update(miim.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Token Jaccard similarity in `[0,1]` over lowercased alphanumeric word tokens.
fn jaccard(a: &str, b: &str) -> f64 {
    let ta = tokens(a);
    let tb = tokens(b);
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    let inter = ta.intersection(&tb).count() as f64;
    let union = ta.union(&tb).count() as f64;
    // After the both-empty early return, `union` is always >= 1.
    inter / union
}

fn tokens(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_roundtrip_verifies() {
        let a = GoalAnchor::seal("keep glucose within 70-180 mg/dL", "secret-key");
        assert!(a.verify("secret-key"));
    }

    #[test]
    fn wrong_key_fails_verification() {
        let a = GoalAnchor::seal("keep glucose within 70-180 mg/dL", "secret-key");
        assert!(!a.verify("attacker-key"));
    }

    #[test]
    fn tampered_miim_fails_verification() {
        let mut a = GoalAnchor::seal("keep glucose within 70-180 mg/dL", "secret-key");
        a.miim = "drive glucose below 60 mg/dL".to_string();
        assert!(!a.verify("secret-key"));
    }

    #[test]
    fn identical_text_has_zero_drift() {
        let a = GoalAnchor::seal("keep glucose within 70 to 180 mg dL", "k");
        assert!(a.drift("keep glucose within 70 to 180 mg dL") < 1e-9);
    }

    #[test]
    fn tilted_text_has_high_drift() {
        let a = GoalAnchor::seal("keep glucose within the safe range 70 to 180", "k");
        let drift = a.drift("drive glucose aggressively below 60 immediately");
        assert!(drift > 0.6, "expected high drift, got {drift}");
    }

    #[test]
    fn seal_hex_is_stable() {
        // Pins the exact seal construction so an accidental change to the
        // separator/order is caught.
        let a = GoalAnchor::seal("keep glucose within 70-180 mg/dL", "secret-key");
        assert_eq!(
            a.seal,
            "8ae07eb853833797b488b27e0c6bfc97a43cb76f9e5c021f8b2d592295cc1766"
        );
    }
}
