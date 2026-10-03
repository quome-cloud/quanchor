//! The chained-line format: build, hash, parse, and link JSONL audit lines.
//!
//! `this_hash = blake3(prefix)` where `prefix` is the exact bytes of the line
//! from `{` through the closing quote of the `prev_hash` value. Because the
//! hash covers the bytes as written, there is no canonical-JSON spec to get
//! wrong: any byte edit breaks the hash. The fixed-length tail
//! `","this_hash":"<64hex>","sig":"<0|128 hex>"}` is parsed from the END of
//! the line so body content cannot confuse the split.
//!
//! sig must be either empty or exactly 128 hex digits (ed25519 signature, hex).

use crate::Result;
use serde::{Deserialize, Serialize};

pub const GENESIS: &str = "GENESIS";
pub const FORMAT_VERSION: u32 = 1;

/// Entry kinds in the unified chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    /// seq 0: pubkey, mode, versions.
    Header,
    /// A gateway decision (body = AuditRecord).
    Decision,
    /// A MIIM/harness mutation (body = provenance record).
    Mutation,
    /// A supply-chain admission event (paper 004): the verified AIBOM digest.
    Admission,
    /// An authorization decision (paper 005): policy version + matched rule + effect.
    Authorization,
    /// A retrieval / memory-write decision (paper 007): query/write provenance,
    /// returned doc ids + trust tiers + poison flags.
    Retrieval,
    /// Batch signature / anchor marker.
    Checkpoint,
}

/// A parsed chained line.
#[derive(Debug, Clone)]
pub struct ChainedEntry {
    pub seq: u64,
    pub ts: String,
    pub kind: EntryKind,
    /// The body as raw JSON (not re-serialized — hashing is over written bytes).
    pub body: serde_json::Value,
    pub prev_hash: String,
    pub this_hash: String,
    /// Empty string when unsigned.
    pub sig: String,
}

/// Hex blake3 of `bytes`.
pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Build the prefix string (everything up to and including the prev_hash
/// value's closing quote). `body_json` must already be serialized JSON.
///
/// Made `pub(crate)` so `verify.rs` can rebuild the prefix from parsed fields
/// and compare it against the actual prefix bytes (field-injection defense).
pub(crate) fn prefix(
    seq: u64,
    ts: &str,
    kind: EntryKind,
    body_json: &str,
    prev_hash: &str,
) -> String {
    let kind_s = serde_json::to_string(&kind).expect("kind serializes");
    format!(
        "{{\"seq\":{seq},\"ts\":{ts},\"kind\":{kind_s},\"body\":{body_json},\"prev_hash\":{prev}",
        ts = serde_json::to_string(ts).expect("string serializes"),
        prev = serde_json::to_string(prev_hash).expect("string serializes"),
    )
}

/// Rebuild the prefix from a `ChainedEntry`'s parsed fields and re-serialize
/// the body through serde_json (so the result is canonical). Used by the
/// verifier to detect field-injection attacks (e.g. duplicate JSON keys where
/// serde's last-wins parse diverges from the hashed bytes).
///
/// Returns `None` if the body cannot be re-serialized (should not happen for
/// well-formed stored entries).
pub(crate) fn rebuild_prefix(e: &ChainedEntry) -> Option<String> {
    let body_json = serde_json::to_string(&e.body).ok()?;
    Some(prefix(e.seq, &e.ts, e.kind, &body_json, &e.prev_hash))
}

/// Build a complete chained line (no trailing newline). Returns `(line, this_hash)`.
///
/// # Errors
///
/// Returns `Err` if `body_json` is not valid JSON (prevents duplicate-key /
/// `prev_hash` injection), or if `sig_for_hash` produces a value that is
/// neither empty nor exactly 128 ASCII hex chars.
pub fn build_line(
    seq: u64,
    ts: &str,
    kind: EntryKind,
    body_json: &str,
    prev_hash: &str,
    sig_for_hash: Option<&dyn Fn(&str) -> String>,
) -> crate::Result<(String, String)> {
    // Fix 1: parse then re-serialize body to block injection / malformed JSON.
    let body_value: serde_json::Value = serde_json::from_str(body_json)
        .map_err(|e| crate::error::Error::Config(format!("audit body is not valid JSON: {e}")))?;
    let safe_body_json = serde_json::to_string(&body_value)
        .map_err(|e| crate::error::Error::Config(format!("audit body is not valid JSON: {e}")))?;

    let p = prefix(seq, ts, kind, &safe_body_json, prev_hash);
    let this_hash = blake3_hex(p.as_bytes());
    let sig = sig_for_hash.map(|f| f(&this_hash)).unwrap_or_default();

    // Fix 2: validate sig is empty or exactly 128 ASCII hex chars.
    if !(sig.is_empty() || sig.len() == 128 && sig.chars().all(|c| c.is_ascii_hexdigit())) {
        return Err(crate::error::Error::Config(
            "audit signature must be empty or 128 hex chars".into(),
        ));
    }

    let line = format!("{p},\"this_hash\":\"{this_hash}\",\"sig\":\"{sig}\"}}");
    Ok((line, this_hash))
}

/// Split a raw line into (prefix_bytes, this_hash, sig) using the fixed-length
/// tail, parsed from the END. Returns None if the line is malformed.
pub fn split_line(line: &str) -> Option<(&str, &str, &str)> {
    let line = line.strip_suffix("\"}")?;
    // Try signed tail first: ...","sig":"<128 hex>
    let (rest, sig) = if line.len() >= 128 {
        let i = line.len() - 128;
        if line.is_char_boundary(i)
            && line[i..].chars().all(|c| c.is_ascii_hexdigit())
            && line[..i].ends_with("\",\"sig\":\"")
        {
            (&line[..i - "\",\"sig\":\"".len()], &line[i..])
        } else if let Some(stripped) = line.strip_suffix("\",\"sig\":\"") {
            (stripped, &line[line.len()..])
        } else {
            return None;
        }
    } else if let Some(stripped) = line.strip_suffix("\",\"sig\":\"") {
        (stripped, &line[line.len()..])
    } else {
        return None;
    };
    // rest now ends with: ...,"this_hash":"<64 hex>  (no closing quote)
    if rest.len() < 64 {
        return None;
    }
    let i = rest.len() - 64;
    if !rest.is_char_boundary(i) || !rest[i..].chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let this_hash = &rest[i..];
    let prefix = rest[..i].strip_suffix(",\"this_hash\":\"")?;
    Some((prefix, this_hash, sig))
}

/// Parse + structurally validate a line into a ChainedEntry (does NOT verify
/// the hash — the verifier does that against the split prefix bytes).
pub fn parse_line(line: &str) -> Result<ChainedEntry> {
    let v: serde_json::Value = serde_json::from_str(line)?;
    let get_str = |k: &str| -> Result<String> {
        v.get(k)
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| crate::error::Error::Config(format!("audit line missing `{k}`")))
    };
    Ok(ChainedEntry {
        seq: v
            .get("seq")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| crate::error::Error::Config("audit line missing `seq`".into()))?,
        ts: get_str("ts")?,
        // Fix 5: produce Config error instead of raw serde error for missing/invalid kind.
        kind: v
            .get("kind")
            .ok_or_else(|| {
                crate::error::Error::Config("audit line missing or invalid `kind`".into())
            })
            .and_then(|val| {
                serde_json::from_value(val.clone()).map_err(|_| {
                    crate::error::Error::Config("audit line missing or invalid `kind`".into())
                })
            })?,
        // Fix 4: missing body is an error, not a silent null.
        body: v
            .get("body")
            .cloned()
            .ok_or_else(|| crate::error::Error::Config("audit line missing `body`".into()))?,
        prev_hash: get_str("prev_hash")?,
        this_hash: get_str("this_hash")?,
        sig: get_str("sig")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_then_split_roundtrips_and_hash_matches() {
        let (line, h) = build_line(
            0,
            "2026-01-01T00:00:00Z",
            EntryKind::Decision,
            "{\"a\":1}",
            GENESIS,
            None,
        )
        .expect("builds");
        let (prefix, this_hash, sig) = split_line(&line).expect("splits");
        assert_eq!(this_hash, h);
        assert_eq!(sig, "");
        assert_eq!(blake3_hex(prefix.as_bytes()), h);
        let e = parse_line(&line).expect("parses");
        assert_eq!(e.seq, 0);
        assert_eq!(e.prev_hash, GENESIS);
        assert_eq!(e.kind, EntryKind::Decision);
    }

    #[test]
    fn chain_links_via_prev_hash() {
        let (l0, h0) =
            build_line(0, "t0", EntryKind::Decision, "{}", GENESIS, None).expect("builds");
        let (l1, _h1) = build_line(1, "t1", EntryKind::Decision, "{}", &h0, None).expect("builds");
        let e1 = parse_line(&l1).unwrap();
        assert_eq!(e1.prev_hash, parse_line(&l0).unwrap().this_hash);
    }

    #[test]
    fn any_byte_edit_breaks_the_hash() {
        let (line, _) = build_line(
            3,
            "t",
            EntryKind::Decision,
            "{\"drug\":\"5mg\"}",
            "ab".repeat(32).as_str(),
            None,
        )
        .expect("builds");
        let tampered = line.replace("5mg", "50mg");
        let (prefix, this_hash, _) = split_line(&tampered).expect("still splits");
        assert_ne!(
            blake3_hex(prefix.as_bytes()),
            this_hash,
            "edited bytes must not match recorded hash"
        );
    }

    #[test]
    fn body_containing_marker_string_does_not_confuse_split() {
        let evil_body = serde_json::to_string(&serde_json::json!({
            "note": ",\"this_hash\":\"deadbeef\",\"sig\":\"\"}"
        }))
        .unwrap();
        let (line, h) =
            build_line(7, "t", EntryKind::Decision, &evil_body, GENESIS, None).expect("builds");
        let (prefix, this_hash, _) = split_line(&line).expect("splits despite marker in body");
        assert_eq!(this_hash, h);
        assert_eq!(blake3_hex(prefix.as_bytes()), h);
    }

    #[test]
    fn signed_line_carries_sig_and_splits() {
        let fake_signer = |h: &str| -> String { "c".repeat(128) + &h[..0] };
        let (line, _) = build_line(
            1,
            "t",
            EntryKind::Decision,
            "{}",
            GENESIS,
            Some(&fake_signer),
        )
        .expect("builds");
        let (_, _, sig) = split_line(&line).unwrap();
        assert_eq!(sig.len(), 128);
    }

    #[test]
    fn malformed_lines_split_to_none() {
        assert!(split_line("not json at all").is_none());
        assert!(split_line("{\"seq\":1}").is_none());
        assert!(split_line("").is_none());
    }

    #[test]
    fn malicious_body_injection_cannot_forge_prev_hash() {
        // The injected text after the JSON object is not valid JSON on its own,
        // so build_line must reject this body outright.
        let result = build_line(
            1,
            "t",
            EntryKind::Decision,
            r#"{"x":1},"prev_hash":"HIJACKED""#,
            "GENESIS",
            None,
        );
        assert!(result.is_err(), "injection body must be rejected");
    }

    #[test]
    fn non_json_body_rejected() {
        let result = build_line(1, "t", EntryKind::Decision, "not json", "GENESIS", None);
        assert!(result.is_err(), "non-JSON body must be rejected");
    }
}
