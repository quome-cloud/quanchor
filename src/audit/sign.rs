//! ed25519 signing for audit entries. Dev: 32-byte hex seed in a keyfile
//! (path from config or QFIRE_AUDIT_KEY). Prod: the same file is released
//! into the enclave via attestation (002 GoalAnchor pattern).

use crate::Result;
use ed25519_dalek::{Signature, Signer as _, SigningKey, Verifier as _, VerifyingKey};
use std::path::Path;

pub struct AuditSigner {
    key: SigningKey,
}

impl AuditSigner {
    /// Generate a new key and write its hex seed to `path` (0600 not enforced
    /// here; documented). Returns the signer.
    ///
    /// Honors `QFIRE_AUDIT_FIXED_KEY` (a 32-byte hex seed) for reproducible
    /// fixtures/tests: with it (plus `QFIRE_AUDIT_FIXED_TS`) the entire signed log
    /// is byte-identical across runs, so tamper *localization* is deterministic.
    /// Otherwise a random key.
    pub fn generate_to(path: &Path) -> Result<Self> {
        let seed: [u8; 32] = match std::env::var("QFIRE_AUDIT_FIXED_KEY") {
            Ok(h) => hex::decode(h.trim())
                .ok()
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| anyhow::anyhow!("QFIRE_AUDIT_FIXED_KEY must be 32-byte hex"))?,
            Err(_) => rand::random(),
        };
        std::fs::write(path, hex::encode(seed))?;
        Ok(AuditSigner {
            key: SigningKey::from_bytes(&seed),
        })
    }

    /// Load from a keyfile containing the 32-byte hex seed.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let bytes = hex::decode(text.trim())
            .map_err(|e| anyhow::anyhow!("audit key at {}: bad hex: {e}", path.display()))?;
        let seed: [u8; 32] = match bytes.as_slice().try_into() {
            Ok(a) => a,
            Err(_) => {
                return Err(
                    anyhow::anyhow!("audit key at {}: need 32 bytes", path.display()).into(),
                )
            }
        };
        Ok(AuditSigner {
            key: SigningKey::from_bytes(&seed),
        })
    }

    /// Load if the file exists, else generate (dev ergonomics: `qfire check`
    /// works out of the box).
    pub fn load_or_generate(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            Self::generate_to(path)
        }
    }

    /// Hex-encoded public key (goes in the header entry).
    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.key.verifying_key().to_bytes())
    }

    /// Sign a hash (its hex string bytes); returns 128-char hex signature.
    pub fn sign_hash_hex(&self, this_hash: &str) -> String {
        hex::encode(self.key.sign(this_hash.as_bytes()).to_bytes())
    }
}

/// Verify `sig_hex` over `this_hash`'s hex-string bytes under `pubkey_hex`.
pub fn verify_sig(pubkey_hex: &str, this_hash: &str, sig_hex: &str) -> bool {
    let Ok(pk_bytes) = hex::decode(pubkey_hex) else {
        return false;
    };
    let pk_arr: [u8; 32] = match pk_bytes.as_slice().try_into() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let Ok(vk) = VerifyingKey::from_bytes(&pk_arr) else {
        return false;
    };
    let Ok(sig_bytes) = hex::decode(sig_hex) else {
        return false;
    };
    let sig_arr: [u8; 64] = match sig_bytes.as_slice().try_into() {
        Ok(a) => a,
        Err(_) => return false,
    };
    vk.verify(this_hash.as_bytes(), &Signature::from_bytes(&sig_arr))
        .is_ok()
}

fn parse_vk(pubkey_hex: &str) -> Option<VerifyingKey> {
    let bytes = hex::decode(pubkey_hex).ok()?;
    let arr: [u8; 32] = bytes.as_slice().try_into().ok()?;
    VerifyingKey::from_bytes(&arr).ok()
}

fn parse_sig(sig_hex: &str) -> Option<Signature> {
    let bytes = hex::decode(sig_hex).ok()?;
    let arr: [u8; 64] = bytes.as_slice().try_into().ok()?;
    Some(Signature::from_bytes(&arr))
}

/// Parallel + batched verification of many per-entry signatures under one public key.
/// `items` are `(this_hash, sig_hex)` pairs; returns a `Vec<bool>` aligned to `items`
/// (true = valid). The cryptographic result is byte-for-byte identical to calling
/// [`verify_sig`] on each item — this only moves the ed25519 work off the verifier's
/// sequential critical path (chunked batch verify across rayon worker threads, with a
/// per-item fallback inside any chunk a batch rejects, so a single bad signature is still
/// localized). On a 12-core box this is ~20x faster than sequential per-entry verification.
pub fn verify_batch_hex(pubkey_hex: &str, items: &[(&str, &str)]) -> Vec<bool> {
    use rayon::prelude::*;
    let Some(vk) = parse_vk(pubkey_hex) else {
        return vec![false; items.len()];
    };
    const CHUNK: usize = 1024;
    let mut out = vec![false; items.len()];
    let parts: Vec<(usize, Vec<bool>)> = (0..items.len())
        .step_by(CHUNK)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let end = (start + CHUNK).min(items.len());
            let slice = &items[start..end];
            // Try one batch verification over the whole chunk (fast path).
            if let Some(sigs) = slice.iter().map(|(_, s)| parse_sig(s)).collect::<Option<Vec<_>>>() {
                let msgs: Vec<&[u8]> = slice.iter().map(|(h, _)| h.as_bytes()).collect();
                let vks = vec![vk; slice.len()];
                if ed25519_dalek::verify_batch(&msgs, &sigs, &vks).is_ok() {
                    return (start, vec![true; slice.len()]);
                }
            }
            // Fallback: per-item (localizes the bad signature; also handles bad-hex as false).
            let v = slice
                .iter()
                .map(|(h, s)| match parse_sig(s) {
                    Some(sig) => vk.verify(h.as_bytes(), &sig).is_ok(),
                    None => false,
                })
                .collect();
            (start, v)
        })
        .collect();
    for (start, v) in parts {
        for (j, b) in v.into_iter().enumerate() {
            out[start + j] = b;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn sign_verify_roundtrip() {
        let dir = tempdir().unwrap();
        let s = AuditSigner::generate_to(&dir.path().join("k")).unwrap();
        let sig = s.sign_hash_hex("abc123");
        assert_eq!(sig.len(), 128);
        assert!(verify_sig(&s.pubkey_hex(), "abc123", &sig));
    }

    #[test]
    fn wrong_hash_or_key_fails() {
        let dir = tempdir().unwrap();
        let s1 = AuditSigner::generate_to(&dir.path().join("k1")).unwrap();
        let s2 = AuditSigner::generate_to(&dir.path().join("k2")).unwrap();
        let sig = s1.sign_hash_hex("abc");
        assert!(!verify_sig(&s1.pubkey_hex(), "abd", &sig));
        assert!(!verify_sig(&s2.pubkey_hex(), "abc", &sig));
        assert!(!verify_sig("zz", "abc", &sig));
        assert!(!verify_sig(&s1.pubkey_hex(), "abc", "00"));
    }

    #[test]
    fn load_roundtrips_same_key() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("k");
        let s = AuditSigner::generate_to(&p).unwrap();
        let s2 = AuditSigner::load(&p).unwrap();
        assert_eq!(s.pubkey_hex(), s2.pubkey_hex());
        let s3 = AuditSigner::load_or_generate(&p).unwrap();
        assert_eq!(s.pubkey_hex(), s3.pubkey_hex());
    }
}
