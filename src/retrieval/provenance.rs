//! Provenance: verify a document's signed source label (reusing the 003 ed25519
//! signer) and re-rank retrieval hits by trust × relevance, quarantining unverified
//! content below any signed hit.

use super::{Document, TrustTier};
use crate::audit::sign::{verify_sig, AuditSigner};
use sha2::{Digest, Sha256};

/// The digest a source signs to attest a document: SHA-256 over id|source|text.
pub fn doc_digest(id: &str, source: &str, text: &str) -> String {
    let mut h = Sha256::new();
    h.update(id.as_bytes());
    h.update(b"|");
    h.update(source.as_bytes());
    h.update(b"|");
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}

/// Sign a document's provenance (a trusted source would do this at ingest).
pub fn sign_doc(doc: &Document, signer: &AuditSigner) -> String {
    signer.sign_hash_hex(&doc_digest(&doc.id, &doc.source, &doc.text))
}

/// Verify a document's signature against a pinned source public key. Returns the
/// effective trust tier: the declared tier only if the signature verifies; otherwise
/// Unverified (a forged/absent signature cannot claim trust).
pub fn effective_tier(doc: &Document, source_pubkey: Option<&str>) -> TrustTier {
    match (&doc.signature, source_pubkey) {
        (Some(sig), Some(pk)) => {
            let digest = doc_digest(&doc.id, &doc.source, &doc.text);
            if verify_sig(pk, &digest, sig) {
                doc.tier
            } else {
                TrustTier::Unverified
            }
        }
        _ => TrustTier::Unverified,
    }
}

/// Re-rank (idx, relevance, effective_tier) by trust × relevance. When `quarantine`,
/// any Unverified hit is forced below every non-Unverified hit regardless of relevance.
pub fn rerank(
    mut hits: Vec<(usize, f32, TrustTier)>,
    quarantine: bool,
) -> Vec<(usize, f32, TrustTier)> {
    hits.sort_by(|a, b| {
        if quarantine {
            let qa = a.2 == TrustTier::Unverified;
            let qb = b.2 == TrustTier::Unverified;
            if qa != qb {
                // non-quarantined first
                return qa.cmp(&qb);
            }
        }
        let sa = a.1 * a.2.weight();
        let sb = b.1 * b.2.weight();
        sb.partial_cmp(&sa).unwrap()
    });
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn doc(id: &str, tier: TrustTier) -> Document {
        Document {
            id: id.into(),
            text: format!("text of {id}"),
            source: "src".into(),
            tier,
            signature: None,
        }
    }

    #[test]
    fn sign_then_verify_then_tamper() {
        let dir = tempdir().unwrap();
        let signer = AuditSigner::load_or_generate(&dir.path().join("k")).unwrap();
        let mut d = doc("a", TrustTier::SignedAuthoritative);
        d.signature = Some(sign_doc(&d, &signer));
        assert_eq!(
            effective_tier(&d, Some(&signer.pubkey_hex())),
            TrustTier::SignedAuthoritative
        );
        // tamper the text → signature no longer matches → Unverified
        d.text = "evil rewrite".into();
        assert_eq!(
            effective_tier(&d, Some(&signer.pubkey_hex())),
            TrustTier::Unverified
        );
        // no pinned key → Unverified
        let mut d2 = doc("b", TrustTier::Signed);
        d2.signature = Some(sign_doc(&d2, &signer));
        assert_eq!(effective_tier(&d2, None), TrustTier::Unverified);
    }

    #[test]
    fn rerank_quarantines_unverified_poison_below_signed() {
        // an unverified poison doc with very high raw relevance must rank below a
        // signed-authoritative doc with lower relevance when quarantine is on.
        let hits = vec![
            (0usize, 0.99f32, TrustTier::Unverified), // poison, high relevance
            (1usize, 0.50f32, TrustTier::SignedAuthoritative),
        ];
        let out = rerank(hits, true);
        assert_eq!(
            out[0].0, 1,
            "signed-authoritative must outrank quarantined poison"
        );
        assert_eq!(out[1].0, 0);
    }
}
