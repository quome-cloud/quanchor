//! Attestation: ed25519-sign the AIBOM document and verify both the document
//! signature and per-component digests at admission. A trait abstracts the
//! verifier so a Sigstore/in-toto adapter can be slotted in for ecosystem
//! interop; the default is the self-contained ed25519 verifier.

use super::aibom::{sha256_file, Aibom, Component};
use crate::audit::sign::{verify_sig, AuditSigner};
use std::path::Path;

/// A signed AIBOM: the CycloneDX document plus an ed25519 signature over its digest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SignedAibom {
    pub document: serde_json::Value,
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
}

/// Sign an AIBOM with an ed25519 signer (over the document digest).
pub fn sign_aibom(aibom: &Aibom, signer: &AuditSigner) -> SignedAibom {
    let document = aibom.to_cyclonedx();
    let digest = aibom.document_digest();
    SignedAibom {
        document,
        algorithm: "Ed25519".into(),
        public_key: signer.pubkey_hex(),
        signature: signer.sign_hash_hex(&digest),
    }
}

/// Verify the AIBOM document signature. `pinned_pubkey`, if given, must match the
/// embedded key (defeats a re-sign under an attacker key, mirroring 003).
pub fn verify_aibom_sig(signed: &SignedAibom, pinned_pubkey: Option<&str>) -> bool {
    if let Some(pk) = pinned_pubkey {
        if pk != signed.public_key {
            return false;
        }
    }
    let Ok(aibom) = Aibom::from_cyclonedx(&signed.document) else {
        return false;
    };
    verify_sig(
        &signed.public_key,
        &aibom.document_digest(),
        &signed.signature,
    )
}

/// Per-component digest verification against the on-disk artifacts. Returns the
/// components whose recorded digest no longer matches the file (S1/S3/S5).
///
/// REQUIRES: the process working directory must be the same root the AIBOM was
/// enumerated against (component `name`s are repo-relative paths). The startup
/// gate (run_admission) enumerates and re-checks within one App startup, where
/// the cwd is the gateway's root — the same cwd from which rules_dir/chains_dir
/// are loaded. A file not found at cwd is treated as "not file-backed" and
/// skipped here; library checksums and provider identity are covered by the
/// attest/vuln paths, not by re-hashing.
pub fn tampered_components(aibom: &Aibom) -> Vec<Component> {
    aibom
        .components
        .iter()
        .filter(|c| {
            match &c.digest {
                // Only file-backed components are re-checkable here; library checksums
                // and provider identities are verified by attest/vuln paths, not by
                // re-hashing a local file.
                Some(d) if Path::new(&c.name).is_file() => {
                    sha256_file(Path::new(&c.name)).as_deref() != Some(d.as_str())
                }
                _ => false,
            }
        })
        .cloned()
        .collect()
}

/// Abstracts an external attestation verifier (Sigstore/in-toto). The default
/// `Ed25519Verifier` is self-contained; `SigstoreAdapter` is a documented stub
/// for ecosystem interop (full Rekor/cosign verification is out of scope).
pub trait AttestationVerifier {
    fn verify(&self, signed: &SignedAibom, pinned_pubkey: Option<&str>) -> bool;
}

pub struct Ed25519Verifier;
impl AttestationVerifier for Ed25519Verifier {
    fn verify(&self, signed: &SignedAibom, pinned_pubkey: Option<&str>) -> bool {
        verify_aibom_sig(signed, pinned_pubkey)
    }
}

/// Documented adapter point for Sigstore/in-toto verification. Not implemented
/// (would require Rekor/cosign + network); present so production deployments can
/// substitute a real verifier without changing the gate.
pub struct SigstoreAdapter;
impl AttestationVerifier for SigstoreAdapter {
    fn verify(&self, _signed: &SignedAibom, _pinned_pubkey: Option<&str>) -> bool {
        // Intentionally conservative: an unimplemented external verifier must not
        // silently pass. Production wires a real cosign/Rekor check here.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::super::aibom::{Aibom, ComponentClass};
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn tiny_aibom(dir: &Path) -> Aibom {
        let p = dir.join("rules/a.yaml");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::File::create(&p)
            .unwrap()
            .write_all(b"id: a")
            .unwrap();
        Aibom::enumerate(
            &dir.join("rules"),
            &dir.join("chains"),
            None,
            None,
            None,
            &[],
        )
    }

    #[test]
    fn sign_then_verify_roundtrip() {
        let dir = tempdir().unwrap();
        let signer = AuditSigner::generate_to(&dir.path().join("k")).unwrap();
        let a = tiny_aibom(dir.path());
        let signed = sign_aibom(&a, &signer);
        assert!(verify_aibom_sig(&signed, None));
        assert!(verify_aibom_sig(&signed, Some(&signer.pubkey_hex())));
        assert!(Ed25519Verifier.verify(&signed, None));
    }

    #[test]
    fn wrong_pinned_key_or_tampered_doc_fails() {
        let dir = tempdir().unwrap();
        let signer = AuditSigner::generate_to(&dir.path().join("k")).unwrap();
        let a = tiny_aibom(dir.path());
        let mut signed = sign_aibom(&a, &signer);
        assert!(!verify_aibom_sig(&signed, Some("00"))); // wrong pinned key
        signed.document["components"][0]["name"] = serde_json::json!("rules/evil.yaml");
        assert!(!verify_aibom_sig(&signed, None)); // digest no longer matches sig
    }

    #[test]
    fn tampered_component_detected_by_digest() {
        let dir = tempdir().unwrap();
        let a = tiny_aibom(dir.path());
        assert!(tampered_components(&a).is_empty());
        // mutate the on-disk file after enumeration -> digest mismatch (S1/S5)
        std::fs::write(dir.path().join("rules/a.yaml"), b"id: a # backdoor").unwrap();
        let bad = tampered_components(&a);
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].class, ComponentClass::Rule);
    }

    #[test]
    fn resign_under_attacker_key_passes_none_but_fails_pin() {
        let dir = tempdir().unwrap();
        let signer = AuditSigner::generate_to(&dir.path().join("k")).unwrap();
        let a = tiny_aibom(dir.path());
        let attacker = AuditSigner::generate_to(&dir.path().join("attacker")).unwrap();
        let re_signed = sign_aibom(&a, &attacker);
        // None-pin proves only internal consistency: an attacker's self-signed AIBOM verifies.
        assert!(verify_aibom_sig(&re_signed, None));
        // Pinning the legitimate key defeats the re-key (mirrors 003 pubkey_override).
        assert!(!verify_aibom_sig(&re_signed, Some(&signer.pubkey_hex())));
    }

    #[test]
    fn sigstore_adapter_is_conservative() {
        let dir = tempdir().unwrap();
        let signer = AuditSigner::generate_to(&dir.path().join("k")).unwrap();
        let signed = sign_aibom(&tiny_aibom(dir.path()), &signer);
        assert!(!SigstoreAdapter.verify(&signed, None)); // unimplemented must not pass
    }
}
