//! A versioned, ed25519-signed policy bundle. The bundle digest (over the sorted
//! policy file contents + version) is signed with the 003 AuditSigner; a tampered
//! or unsigned bundle fails verification. The decision log records the bundle
//! version so an auditor can tie a decision to an exact policy revision.

use crate::audit::sign::{verify_sig, AuditSigner};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BundleManifest {
    pub version: String,
    /// filename -> SHA-256 hex of its contents, sorted by filename.
    pub files: Vec<(String, String)>,
    pub public_key: String,
    /// ed25519 over the bundle digest.
    pub signature: String,
}

/// Compute the bundle digest: SHA-256 over `version` then each `name:hash` line
/// in sorted order. Deterministic and order-independent of directory listing.
pub fn bundle_digest(version: &str, files: &[(String, String)]) -> String {
    let mut sorted = files.to_vec();
    sorted.sort();
    let mut h = Sha256::new();
    h.update(version.as_bytes());
    for (name, hash) in &sorted {
        h.update(b"\n");
        h.update(name.as_bytes());
        h.update(b":");
        h.update(hash.as_bytes());
    }
    hex::encode(h.finalize())
}

fn sha256_file(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|b| {
        let mut h = Sha256::new();
        h.update(&b);
        hex::encode(h.finalize())
    })
}

/// Enumerate *.cedar and *.rego files under `dir` as (filename, sha256) pairs.
fn enumerate_policies(dir: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(dir).into_iter().flatten() {
        let p = entry.path();
        if p.is_file() {
            let ext_ok = matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("cedar") | Some("rego")
            );
            if ext_ok {
                if let Some(d) = sha256_file(p) {
                    let name = p.strip_prefix(dir).unwrap_or(p).to_string_lossy().to_string();
                    out.push((name, d));
                }
            }
        }
    }
    out.sort();
    out
}

/// Sign a bundle: enumerate policies under `dir`, compute the digest, sign it.
pub fn sign_bundle(dir: &Path, version: &str, signer: &AuditSigner) -> BundleManifest {
    let files = enumerate_policies(dir);
    let digest = bundle_digest(version, &files);
    BundleManifest {
        version: version.to_string(),
        files,
        public_key: signer.pubkey_hex(),
        signature: signer.sign_hash_hex(&digest),
    }
}

/// Verify a bundle manifest against the on-disk policies and the signature.
/// `pinned_pubkey`, if given, must match the manifest key (defeats a re-sign
/// under an attacker key). With `None`, verification proves only internal
/// consistency (the files match a self-signed manifest), NOT authenticity — an
/// attacker who can rewrite the files AND the manifest can self-sign and pass.
/// Production callers MUST pass the pinned key.
pub fn verify_bundle(dir: &Path, manifest: &BundleManifest, pinned_pubkey: Option<&str>) -> bool {
    if let Some(pk) = pinned_pubkey {
        if pk != manifest.public_key {
            return false;
        }
    }
    // recompute file hashes and confirm they match the manifest
    let on_disk = enumerate_policies(dir);
    if on_disk != manifest.files {
        return false;
    }
    let digest = bundle_digest(&manifest.version, &manifest.files);
    verify_sig(&manifest.public_key, &digest, &manifest.signature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write(dir: &Path, name: &str, body: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    #[test]
    fn sign_then_verify_roundtrip() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "a.cedar",
            "permit(principal, action, resource);",
        );
        write(dir.path(), "b.rego", "package authz\nallow := true");
        let signer = AuditSigner::generate_to(&dir.path().join("k")).unwrap();
        let m = sign_bundle(dir.path(), "v1", &signer);
        assert!(verify_bundle(dir.path(), &m, None));
        assert!(verify_bundle(dir.path(), &m, Some(&signer.pubkey_hex())));
    }

    #[test]
    fn tampered_policy_or_wrong_key_fails() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "a.cedar",
            "permit(principal, action, resource);",
        );
        let signer = AuditSigner::generate_to(&dir.path().join("k")).unwrap();
        let m = sign_bundle(dir.path(), "v1", &signer);
        // tamper a policy file after signing
        write(
            dir.path(),
            "a.cedar",
            "permit(principal, action, resource); // evil",
        );
        assert!(!verify_bundle(dir.path(), &m, None));
        // wrong pinned key
        let m2 = sign_bundle(dir.path(), "v1", &signer);
        assert!(!verify_bundle(dir.path(), &m2, Some("00")));
    }
}
