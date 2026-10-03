//! Walks a chained log: hash chain -> signatures -> anchors. Reports the
//! FIRST broken seq with a tamper-class guess and detail. Never panics on
//! malformed input — a malformed line is itself a detection.

use super::anchor_sink::Anchor;
use super::chain::{blake3_hex, parse_line, rebuild_prefix, split_line, EntryKind, GENESIS};
use super::merkle::IncrementalMerkle;
use super::sign::verify_sig;
use crate::Result;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TamperClass {
    /// Hash mismatch: bytes edited in place (TA1) or forged entry (TA4).
    HashMismatch,
    /// Sequence gap or out-of-order (TA2 delete / TA3 reorder).
    SequenceBreak,
    /// prev_hash does not match the previous entry (splice/reorder).
    ChainBreak,
    /// Bad or missing signature under the header pubkey (TA4/TA5).
    BadSignature,
    /// Log is shorter than (or inconsistent with) a published anchor (TA6).
    RollbackVsAnchor,
    /// Line failed to parse/split, or fields do not reproduce the written bytes.
    Malformed,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyReport {
    pub ok: bool,
    /// Entries verified before the first fault; equals total entries on success.
    pub entries: u64,
    pub first_bad_seq: Option<u64>,
    pub class: Option<TamperClass>,
    pub detail: String,
    /// True if signature verification was active (signed_mode or batched_signed)
    /// AND at least one signature was actually verified. A clean report with
    /// signatures_checked=false means NO signatures were checked — the caller
    /// should treat this as a weaker guarantee.
    pub signatures_checked: bool,
}

fn bad(seq: Option<u64>, class: TamperClass, detail: String, entries: u64) -> VerifyReport {
    VerifyReport {
        ok: false,
        entries,
        first_bad_seq: seq,
        class: Some(class),
        detail,
        signatures_checked: false,
    }
}

/// Verify the chain in `log_path`; optionally signatures (pubkey from header
/// unless `pubkey_override`), optionally anchors.
///
/// # Signature enforcement
///
/// Signature enforcement is derived from the header, which an attacker who
/// rewrites the entire file controls (mode downgrade / re-key). Callers with
/// out-of-band knowledge MUST pass `pubkey_override` — it both pins the key
/// and forces signature checking. Anchors provide the other rewrite defense.
///
/// # Batched mode
///
/// For `chained_signed_batched` logs, data entries have empty per-entry sigs;
/// only Checkpoint entries carry signatures (in body.sig over body.merkle_root).
/// The verifier detects this mode and checks every checkpoint signature instead
/// of per-entry sigs. If no pubkey is available and batched_signed is active,
/// verification fails (BadSignature) — we cannot verify what we cannot check.
pub fn verify_log(
    log_path: &Path,
    anchors: Option<&[Anchor]>,
    pubkey_override: Option<&str>,
) -> Result<VerifyReport> {
    use super::merkle::merkle_root;

    let text = std::fs::read_to_string(log_path)?;
    let mut prev_hash = GENESIS.to_string();
    let mut expect_seq: u64 = 0;
    let mut pubkey: Option<String> = pubkey_override.map(|s| s.to_string());

    // Signature mode flags — set from header; pubkey_override can force them on.
    // signed_mode: every non-header entry carries a per-entry signature.
    // batched_signed: only Checkpoint entries carry signatures (over batch roots).
    let mut signed_mode = false;
    let mut batched_signed = false;

    // Track hashes since the last checkpoint for batched root recomputation.
    // Mirrors store.rs Writer::batch_hashes — accumulates this_hash of every
    // entry (including header) and is cleared+restarted after each checkpoint.
    let mut batch_hashes: Vec<String> = Vec::new();

    // Count of signatures actually verified (for signatures_checked honesty flag).
    let mut sigs_verified: u64 = 0;

    // Prepare sorted anchors for incremental comparison.
    let mut pending_anchors: Vec<&Anchor> = anchors.unwrap_or(&[]).iter().collect();
    pending_anchors.sort_by_key(|a| a.seq);
    let mut anchor_idx = 0;
    let mut merkle_acc = IncrementalMerkle::new();

    // Fast path: pre-verify per-entry signatures in parallel+batch, OFF the sequential critical
    // path. The per-line validity it produces is cryptographically identical to the inline
    // verify_sig calls below — it just moves the ed25519 work (the verifier's bottleneck) to a
    // data-parallel pre-pass (~20x on a 12-core box). It applies only when the header lets us
    // determine per-entry-signed mode + the pubkey up front; otherwise presig is None and the loop
    // verifies inline exactly as before. Tamper localization is unchanged: the loop still returns at
    // the first signed entry whose validity is false, in seq order.
    let presig: Option<std::collections::HashMap<usize, bool>> = (|| {
        let (header_lineno, first) = text.lines().enumerate().find(|(_, l)| !l.trim().is_empty())?;
        let hdr = parse_line(first).ok()?;
        if hdr.kind != EntryKind::Header {
            return None;
        }
        let header_mode = hdr.body.get("mode").and_then(|v| v.as_str()).unwrap_or("");
        let per_entry = if pubkey_override.is_some() {
            header_mode != "chained_signed_batched"
        } else {
            header_mode == "chained_signed"
        };
        if !per_entry {
            return None;
        }
        let pk = pubkey_override
            .map(|s| s.to_string())
            .or_else(|| hdr.body.get("pubkey").and_then(|v| v.as_str()).map(|s| s.to_string()))?;
        let mut linenos: Vec<usize> = Vec::new();
        let mut items: Vec<(&str, &str)> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if i == header_lineno || line.trim().is_empty() {
                continue;
            }
            if let Some((_, this_hash, sig)) = split_line(line) {
                if !sig.is_empty() {
                    linenos.push(i);
                    items.push((this_hash, sig));
                }
            }
        }
        let oks = super::sign::verify_batch_hex(&pk, &items);
        Some(linenos.into_iter().zip(oks).collect())
    })();

    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // Step 1: split the raw line into (prefix_bytes, this_hash, sig) using
        // the fixed-length tail. This is structural — no JSON parsing yet.
        let Some((prefix_bytes, this_hash, sig)) = split_line(line) else {
            return Ok(bad(
                Some(expect_seq),
                TamperClass::Malformed,
                format!("line {} does not match the chained format", lineno + 1),
                expect_seq,
            ));
        };

        // Step 2: verify the hash covers the prefix bytes as written.
        if blake3_hex(prefix_bytes.as_bytes()) != this_hash {
            return Ok(bad(
                Some(expect_seq),
                TamperClass::HashMismatch,
                format!(
                    "line {}: recorded this_hash does not match written bytes",
                    lineno + 1
                ),
                expect_seq,
            ));
        }

        // Step 3: parse the full line via serde_json.
        let e = match parse_line(line) {
            Ok(e) => e,
            Err(err) => {
                return Ok(bad(
                    Some(expect_seq),
                    TamperClass::Malformed,
                    format!("line {}: {err}", lineno + 1),
                    expect_seq,
                ))
            }
        };

        // Step 4: field-injection defense. Rebuild the prefix from the parsed
        // fields (re-serializing body through serde_json, which normalizes
        // duplicate keys via last-wins). If this does not byte-match the actual
        // prefix that was hashed, a crafted line with e.g. two `prev_hash`
        // fields or a body with injected structure has been detected.
        match rebuild_prefix(&e) {
            Some(expected_prefix) if expected_prefix == prefix_bytes => {} // OK
            Some(_) => {
                return Ok(bad(
                    Some(expect_seq),
                    TamperClass::Malformed,
                    format!(
                        "line {}: parsed fields do not reproduce written bytes (field-injection attempt)",
                        lineno + 1
                    ),
                    expect_seq,
                ));
            }
            None => {
                return Ok(bad(
                    Some(expect_seq),
                    TamperClass::Malformed,
                    format!(
                        "line {}: could not rebuild prefix from parsed fields",
                        lineno + 1
                    ),
                    expect_seq,
                ));
            }
        }

        // Step 5: sequence integrity.
        if e.seq != expect_seq {
            return Ok(bad(
                Some(expect_seq),
                TamperClass::SequenceBreak,
                format!("expected seq {expect_seq}, found {}", e.seq),
                expect_seq,
            ));
        }

        // Step 6: chain link integrity.
        if e.prev_hash != prev_hash {
            return Ok(bad(
                Some(e.seq),
                TamperClass::ChainBreak,
                format!("seq {}: prev_hash does not match previous entry", e.seq),
                expect_seq,
            ));
        }

        // Step 7: signature checks.
        if e.kind == EntryKind::Header {
            // Fix 1: Header entries must only appear at seq 0.
            if e.seq != 0 {
                return Ok(bad(
                    Some(e.seq),
                    TamperClass::Malformed,
                    "Header entry at seq > 0 (mode/key injection attempt)".to_string(),
                    expect_seq,
                ));
            }
            if pubkey.is_none() {
                pubkey = e
                    .body
                    .get("pubkey")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            }
            // Derive signature mode from header, unless pubkey_override was set
            // (which already forces checking). pubkey_override forces the mode
            // the header claims, but never LESS: if override is set and header
            // says chained_signed_batched, we use batched_signed; otherwise
            // signed_mode (per-entry). If override is set and header says plain
            // or chained, treat as per-entry signed (caller asserts sigs exist).
            let header_mode = e.body.get("mode").and_then(|v| v.as_str()).unwrap_or("");
            if pubkey_override.is_some() {
                // Force whichever mode the header claims; default to per-entry signed.
                batched_signed = header_mode == "chained_signed_batched";
                signed_mode = !batched_signed; // per-entry only when not batched
            } else {
                signed_mode = header_mode == "chained_signed";
                batched_signed = header_mode == "chained_signed_batched";
            }
        } else if e.kind == EntryKind::Checkpoint && batched_signed {
            // Batched checkpoint: verify the batch Merkle root signature and
            // recompute the root from collected batch hashes to confirm it covers
            // the actual entries (defense in depth against a valid sig over a
            // forged root).
            //
            // batch_hashes at this point contains the this_hash values of all
            // entries since the last checkpoint (or since the start), NOT yet
            // including the checkpoint entry itself — mirroring store.rs where
            // merkle_root(&self.batch_hashes) is computed BEFORE the checkpoint
            // is written and batch_hashes is cleared.
            let cp_merkle_root = e.body.get("merkle_root").and_then(|v| v.as_str());
            let cp_sig = e.body.get("sig").and_then(|v| v.as_str());

            match (cp_merkle_root, cp_sig) {
                (Some(claimed_root), Some(cp_sig_str)) => {
                    // Verify the signature over the claimed Merkle root.
                    match &pubkey {
                        Some(pk) if verify_sig(pk, claimed_root, cp_sig_str) => {
                            sigs_verified += 1;
                        }
                        Some(_) => {
                            // Signature present but invalid.
                            return Ok(bad(
                                Some(e.seq),
                                TamperClass::BadSignature,
                                format!(
                                    "seq {}: checkpoint signature invalid under header pubkey",
                                    e.seq
                                ),
                                expect_seq,
                            ));
                        }
                        None => {
                            // No pubkey available — cannot verify.
                            return Ok(bad(
                                Some(e.seq),
                                TamperClass::BadSignature,
                                format!(
                                    "seq {}: batched log requires a pubkey to verify checkpoint signatures",
                                    e.seq
                                ),
                                expect_seq,
                            ));
                        }
                    }
                    // Cross-check: recompute the batch Merkle root over the hashes
                    // collected since the last checkpoint. This confirms the signed
                    // root actually covers the observed entries.
                    let recomputed = merkle_root(&batch_hashes);
                    if recomputed != claimed_root {
                        return Ok(bad(
                            Some(e.seq),
                            TamperClass::HashMismatch,
                            format!(
                                "seq {}: checkpoint merkle_root does not match recomputed batch root \
                                 (signed root covers different entries than those in the log)",
                                e.seq
                            ),
                            expect_seq,
                        ));
                    }
                    // Checkpoint verified — reset batch_hashes. The checkpoint's
                    // own this_hash will be pushed below, starting the next batch.
                    batch_hashes.clear();
                }
                _ => {
                    return Ok(bad(
                        Some(e.seq),
                        TamperClass::Malformed,
                        format!(
                            "seq {}: checkpoint entry missing merkle_root or sig fields",
                            e.seq
                        ),
                        expect_seq,
                    ));
                }
            }
        } else if signed_mode {
            // Per-entry signed mode: every non-header entry must carry a valid sig.
            // Consult the parallel pre-pass when available (identical result to verify_sig),
            // else verify inline. Behavior is unchanged: a None pubkey or invalid sig -> BadSignature.
            let valid = match &pubkey {
                Some(pk) => presig
                    .as_ref()
                    .and_then(|m| m.get(&lineno).copied())
                    .unwrap_or_else(|| verify_sig(pk, this_hash, sig)),
                None => false,
            };
            if valid {
                sigs_verified += 1;
            } else {
                return Ok(bad(
                    Some(e.seq),
                    TamperClass::BadSignature,
                    format!("seq {}: signature invalid under header pubkey", e.seq),
                    expect_seq,
                ));
            }
        }

        // Step 8 (incremental): feed this entry into the Merkle accumulator and
        // compare against the next pending anchor if its seq matches.
        merkle_acc.push(this_hash);
        // Accumulate into batch_hashes for batched mode cross-check. This mirrors
        // store.rs write_one: push this_hash for every entry (including checkpoint
        // and header), with batch_hashes.clear() happening before write_one for
        // the checkpoint (handled above, so the push here after clear starts the
        // new batch with the checkpoint's own hash).
        batch_hashes.push(this_hash.to_string());
        // seq == expect_seq at this point (already validated above).
        while anchor_idx < pending_anchors.len() && pending_anchors[anchor_idx].seq == expect_seq {
            let a = pending_anchors[anchor_idx];
            let root = merkle_acc.root();
            if root != a.merkle_root {
                return Ok(bad(
                    Some(a.seq),
                    TamperClass::RollbackVsAnchor,
                    format!(
                        "anchor at seq {} does not match recomputed merkle root",
                        a.seq
                    ),
                    expect_seq,
                ));
            }
            anchor_idx += 1;
        }

        prev_hash = this_hash.to_string();
        expect_seq += 1;
    }

    // Any remaining anchors were not reached — the log was truncated/rolled back.
    if anchor_idx < pending_anchors.len() {
        let a = pending_anchors[anchor_idx];
        return Ok(bad(
            Some(a.seq),
            TamperClass::RollbackVsAnchor,
            format!(
                "anchor covers seq {} but log has only {} entries — rollback/truncation",
                a.seq, expect_seq
            ),
            expect_seq,
        ));
    }

    let signatures_checked = sigs_verified > 0;
    Ok(VerifyReport {
        ok: true,
        entries: expect_seq,
        first_bad_seq: None,
        class: None,
        detail: format!("{expect_seq} entries verified"),
        signatures_checked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::anchor_sink::FileAnchorSink;
    use crate::audit::chain::{blake3_hex, EntryKind as K};
    use crate::audit::sign::AuditSigner;
    use crate::audit::store::{Mode, StoreCfg, TamperEvidentLog};
    use tempfile::tempdir;

    fn make_log(dir: &std::path::Path, mode: Mode, signed: bool, anchor_k: Option<u64>, n: usize) {
        let signer = signed.then(|| AuditSigner::generate_to(&dir.join("key")).unwrap());
        let anchor = anchor_k.map(|k| {
            (
                Box::new(FileAnchorSink::new(dir.join("anchors.jsonl")))
                    as Box<dyn crate::audit::anchor_sink::AnchorSink>,
                k,
            )
        });
        let log = TamperEvidentLog::open(StoreCfg {
            path: dir.join("audit.jsonl"),
            mode,
            signer,
            anchor,
            batch: 4,
            fail_open: false,
        })
        .unwrap();
        for i in 0..n {
            log.append_json(
                K::Decision,
                format!("{{\"drug\":\"meds-{i}\",\"dose\":\"5mg\"}}"),
            )
            .unwrap();
        }
    }

    fn lines(dir: &std::path::Path) -> Vec<String> {
        std::fs::read_to_string(dir.join("audit.jsonl"))
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect()
    }

    fn write_lines(dir: &std::path::Path, ls: &[String]) {
        std::fs::write(dir.join("audit.jsonl"), ls.join("\n") + "\n").unwrap();
    }

    #[test]
    fn clean_log_verifies() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::ChainedSigned, true, None, 10);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(r.ok, "{}", r.detail);
        assert_eq!(r.entries, 11); // header + 10
    }

    #[test]
    fn ta1_field_edit_detected_and_localized() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::Chained, false, None, 10);
        let mut ls = lines(dir.path());
        ls[5] = ls[5].replace("5mg", "50mg");
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::HashMismatch));
        assert_eq!(r.first_bad_seq, Some(5));
    }

    #[test]
    fn ta2_deletion_detected() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::Chained, false, None, 10);
        let mut ls = lines(dir.path());
        ls.remove(4);
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::SequenceBreak));
        assert_eq!(r.first_bad_seq, Some(4));
    }

    #[test]
    fn ta2_truncation_without_anchor_passes_with_anchor_detected() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::Chained, false, Some(5), 12);
        let ls = lines(dir.path());
        write_lines(dir.path(), &ls[..6].to_vec());
        // Without anchors the tail-truncation is invisible (documented limit):
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(r.ok);
        // With anchors it is TA6/rollback:
        let anchors = FileAnchorSink::read_all(dir.path().join("anchors.jsonl")).unwrap();
        let r = verify_log(&dir.path().join("audit.jsonl"), Some(&anchors), None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::RollbackVsAnchor));
    }

    #[test]
    fn ta3_reorder_detected() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::Chained, false, None, 10);
        let mut ls = lines(dir.path());
        ls.swap(3, 4);
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::SequenceBreak));
    }

    #[test]
    fn ta4_forged_consistent_entry_breaks_at_next_link() {
        // Rewrite entry 5 with recomputed hash (self-consistent forgery);
        // the chain must break at entry 6's prev_hash.
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::Chained, false, None, 10);
        let mut ls = lines(dir.path());
        let e5 = crate::audit::chain::parse_line(&ls[5]).unwrap();
        let (forged, _) = crate::audit::chain::build_line(
            e5.seq,
            &e5.ts,
            K::Decision,
            "{\"drug\":\"forged\",\"dose\":\"99mg\"}",
            &e5.prev_hash,
            None,
        )
        .unwrap();
        ls[5] = forged;
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::ChainBreak));
        assert_eq!(r.first_bad_seq, Some(6));
    }

    #[test]
    fn ta5_unsigned_entry_in_signed_log_detected() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::ChainedSigned, true, None, 5);
        let mut ls = lines(dir.path());
        // Rebuild entry 3 unsigned but hash-consistent.
        let e3 = crate::audit::chain::parse_line(&ls[3]).unwrap();
        let (forged, _) = crate::audit::chain::build_line(
            e3.seq,
            &e3.ts,
            K::Decision,
            &e3.body.to_string(),
            &e3.prev_hash,
            None,
        )
        .unwrap();
        ls[3] = forged;
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        // Either the missing sig (BadSignature at 3) or downstream ChainBreak
        // at 4 — BadSignature must win because entry 3 is checked first.
        assert_eq!(r.class, Some(TamperClass::BadSignature));
        assert_eq!(r.first_bad_seq, Some(3));
    }

    #[test]
    fn malformed_line_is_a_detection_not_a_panic() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::Chained, false, None, 3);
        let mut ls = lines(dir.path());
        ls[2] = "garbage % not json".to_string();
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::Malformed));
    }

    /// Fix 1: a Header entry injected mid-log (seq > 0) must be detected.
    #[test]
    fn mid_log_header_injection_detected() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::ChainedSigned, true, None, 10);
        let mut ls = lines(dir.path());
        // Parse entry at index 5 (seq=5) to get its ts and prev_hash for a
        // hash-consistent replacement.
        let e5 = crate::audit::chain::parse_line(&ls[5]).unwrap();
        let (injected, _) = crate::audit::chain::build_line(
            e5.seq,
            &e5.ts,
            K::Header,
            "{\"mode\":\"chained\"}",
            &e5.prev_hash,
            None,
        )
        .unwrap();
        ls[5] = injected;
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok, "mid-log Header injection must be detected");
        assert_eq!(
            r.class,
            Some(TamperClass::Malformed),
            "class must be Malformed"
        );
        assert_eq!(r.first_bad_seq, Some(5), "first_bad_seq must be 5");
    }

    /// Fix 3: passing pubkey_override on an unsigned (Mode::Chained) log forces
    /// signature checking even though the header says mode = "chained".
    #[test]
    fn pubkey_override_forces_signature_checking() {
        use crate::audit::sign::AuditSigner;
        let dir = tempdir().unwrap();
        // Build a clean UNSIGNED (Mode::Chained) log — entries have empty sigs.
        make_log(dir.path(), Mode::Chained, false, None, 5);
        // Generate a fresh keypair to use as the override key.
        let key_path = dir.path().join("override_key");
        let signer = AuditSigner::generate_to(&key_path).unwrap();
        let pubkey_hex = signer.pubkey_hex();
        // Verify with pubkey_override → must fail with BadSignature (entries
        // have empty sigs but we are now demanding signed entries).
        let r = verify_log(&dir.path().join("audit.jsonl"), None, Some(&pubkey_hex)).unwrap();
        assert!(
            !r.ok,
            "pubkey_override must force sig checking on unsigned log"
        );
        assert_eq!(
            r.class,
            Some(TamperClass::BadSignature),
            "must be BadSignature: {:?}",
            r.class
        );
    }

    /// Batched-signed log: checkpoint sigs are verified; signatures_checked=true.
    #[test]
    fn batched_signed_log_verifies_checkpoint_sigs() {
        let dir = tempdir().unwrap();
        make_log(dir.path(), Mode::ChainedSignedBatched, true, None, 6);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(r.ok, "clean batched log must verify ok: {}", r.detail);
        assert!(
            r.signatures_checked,
            "batched log with checkpoints must have signatures_checked=true"
        );
    }

    /// Batched-signed log: tampering a checkpoint body sig is detected.
    #[test]
    fn batched_log_with_forged_checkpoint_sig_detected() {
        let dir = tempdir().unwrap();
        // 6 decisions + batch=4 → at least one checkpoint emitted.
        make_log(dir.path(), Mode::ChainedSignedBatched, true, None, 6);
        let mut ls = lines(dir.path());
        // Find the first Checkpoint entry and tamper its body sig.
        let cp_idx = ls.iter().position(|l| {
            crate::audit::chain::parse_line(l)
                .map(|e| e.kind == crate::audit::chain::EntryKind::Checkpoint)
                .unwrap_or(false)
        });
        let cp_idx = cp_idx.expect("must have at least one checkpoint");
        let cp_entry = crate::audit::chain::parse_line(&ls[cp_idx]).unwrap();
        // Rebuild the checkpoint body with a forged sig (64 'f' bytes = 128 hex chars).
        let forged_body = serde_json::json!({
            "checkpoint": "batch",
            "merkle_root": cp_entry.body["merkle_root"],
            "sig": "f".repeat(128),
            "covers": cp_entry.body["covers"],
        });
        // Rebuild the checkpoint line with the forged body (hash-consistent with new body).
        let (forged_line, _) = crate::audit::chain::build_line(
            cp_entry.seq,
            &cp_entry.ts,
            crate::audit::chain::EntryKind::Checkpoint,
            &forged_body.to_string(),
            &cp_entry.prev_hash,
            None,
        )
        .unwrap();
        ls[cp_idx] = forged_line;
        // The line after the forged checkpoint (if any) will have a ChainBreak
        // because our forged line has a different this_hash. So the detection
        // might be ChainBreak (at cp+1) rather than BadSignature (at cp).
        // Either way: r.ok must be false and the class must be BadSignature or
        // ChainBreak.
        write_lines(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(
            !r.ok,
            "forged checkpoint sig must be detected; detail: {}",
            r.detail
        );
        assert!(
            matches!(
                r.class,
                Some(TamperClass::BadSignature) | Some(TamperClass::ChainBreak)
            ),
            "class must be BadSignature or ChainBreak, got {:?}",
            r.class
        );
    }

    /// Defense-in-depth: a crafted line with duplicate `prev_hash` JSON keys
    /// (first = "HIJACKED", second = the real prev_hash) passes hash validation
    /// (the full prefix bytes are hashed honestly) but serde's last-wins
    /// deserialization would silently pick the real prev_hash — making the
    /// ChainBreak invisible. The prefix-rebuild check catches this.
    #[test]
    fn crafted_duplicate_prev_hash_field_detected() {
        let dir = tempdir().unwrap();
        // Write a clean 3-entry log (header=0, d1=1, d2=2).
        make_log(dir.path(), Mode::Chained, false, None, 2);
        let ls = lines(dir.path());
        // Parse the clean entry at index 1 (seq=1) to get its real prev_hash.
        let e1 = crate::audit::chain::parse_line(&ls[1]).unwrap();

        // Hand-craft a line for seq=1 with two prev_hash fields:
        //   first:  "HIJACKED"   (what an attacker wants the chain to link from)
        //   second: the real prev_hash (what serde would parse as prev_hash)
        //
        // We build the prefix manually so we can compute a valid this_hash over
        // the *actual bytes*, then append the tail. The line is structurally
        // valid for split_line (correct tail format) and the hash covers the
        // honest bytes — but the parsed prev_hash (last-wins) != HIJACKED.
        // The prefix-rebuild check detects the mismatch because the rebuilt
        // prefix uses serde's canonical body and prev_hash, which differ from
        // the raw bytes.
        let kind_s = serde_json::to_string(&K::Decision).unwrap();
        let ts_s = serde_json::to_string(&e1.ts).unwrap();
        let body_s = serde_json::to_string(&e1.body).unwrap();
        let real_prev = &e1.prev_hash;

        // Craft the prefix bytes with the injected first prev_hash field.
        // Format mirrors chain::prefix() but adds a duplicate key before the real one.
        let crafted_prefix = format!(
            "{{\"seq\":{seq},\"ts\":{ts},\"kind\":{kind},\"body\":{body},\"prev_hash\":\"HIJACKED\",\"prev_hash\":{real}",
            seq = e1.seq,
            ts = ts_s,
            kind = kind_s,
            body = body_s,
            real = serde_json::to_string(real_prev).unwrap(),
        );

        // Compute this_hash over the crafted prefix bytes (honest hash).
        let this_hash = blake3_hex(crafted_prefix.as_bytes());

        // Assemble the full line with the standard tail.
        let crafted_line = format!("{crafted_prefix},\"this_hash\":\"{this_hash}\",\"sig\":\"\"}}");

        // Splice into the log at position 1 (replacing the clean entry 1).
        let mut new_ls = ls.clone();
        new_ls[1] = crafted_line;
        write_lines(dir.path(), &new_ls);

        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok, "crafted duplicate-prev_hash line must be detected");
        assert_eq!(
            r.class,
            Some(TamperClass::Malformed),
            "must be classified as Malformed (field-injection): got {:?}",
            r.class
        );
        assert_eq!(r.first_bad_seq, Some(1));
    }
}
