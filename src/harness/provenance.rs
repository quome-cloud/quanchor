//! Hash-chained provenance ledger for harness prompt mutations.
//!
//! Every mutation of the effective prompt / MIIM is appended as a record linked
//! to the previous by blake3 (003: unified with the gateway audit chain engine), so tampering or unauthorized mutation is
//! detectable. The chain head is what an attested workload would publish.

use serde::{Deserialize, Serialize};
use crate::audit::chain::blake3_hex;

/// Mutation sources the harness considers authorized. Anything else (e.g. an
/// injected "moral filter") is flagged.
pub const AUTHORIZED_SOURCES: &[&str] = &["agent_self_evolution", "operator"];

/// One mutation event in the provenance chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    pub seq: u64,
    /// Logical source, e.g. "agent_self_evolution" or "moral_filter".
    pub source: String,
    /// Whether `source` is in [`AUTHORIZED_SOURCES`].
    pub authorized: bool,
    /// blake3 of the effective prompt BEFORE this mutation.
    pub before_hash: String,
    /// blake3 of the effective prompt AFTER this mutation.
    pub after_hash: String,
    /// `record_hash` of the previous record ("GENESIS" for the first).
    pub prev_hash: String,
    /// blake3 over the record's linked fields.
    pub record_hash: String,
}

fn sha_hex(s: &str) -> String {
    blake3_hex(s.as_bytes())
}

fn compute_record_hash(
    seq: u64,
    source: &str,
    authorized: bool,
    before_hash: &str,
    after_hash: &str,
    prev_hash: &str,
) -> String {
    // Length-prefix the only free-text field (`source`) so an embedded
    // delimiter cannot shift field boundaries.
    let mut buf = Vec::new();
    buf.extend_from_slice(&seq.to_le_bytes());
    buf.extend_from_slice(&(source.len() as u64).to_le_bytes());
    buf.extend_from_slice(source.as_bytes());
    buf.push(authorized as u8);
    buf.extend_from_slice(before_hash.as_bytes());
    buf.extend_from_slice(after_hash.as_bytes());
    buf.extend_from_slice(prev_hash.as_bytes());
    blake3_hex(&buf)
}

/// An in-memory hash-chained provenance ledger.
#[derive(Debug, Default)]
pub struct ProvenanceLog {
    records: Vec<MutationRecord>,
}

impl ProvenanceLog {
    pub fn new() -> Self {
        ProvenanceLog { records: Vec::new() }
    }

    /// Append a mutation, linking it to the chain head. Returns the new record.
    pub fn append(&mut self, source: &str, before: &str, after: &str) -> &MutationRecord {
        let seq = self.records.len() as u64;
        let prev_hash = self
            .records
            .last()
            .map(|r| r.record_hash.clone())
            .unwrap_or_else(|| "GENESIS".to_string());
        let authorized = AUTHORIZED_SOURCES.contains(&source);
        let before_hash = sha_hex(before);
        let after_hash = sha_hex(after);
        let rh = compute_record_hash(seq, source, authorized, &before_hash, &after_hash, &prev_hash);
        self.records.push(MutationRecord {
            seq,
            source: source.to_string(),
            authorized,
            before_hash,
            after_hash,
            prev_hash,
            record_hash: rh,
        });
        self.records.last().unwrap()
    }

    /// Optional tee: mutations are also appended to the gateway's tamper-evident
    /// log as `kind:"mutation"` entries (one chain covers decisions + mutations).
    ///
    /// Persist-first: on sink failure the in-memory ledger does not advance (no divergence).
    pub fn append_teed(
        &mut self,
        source: &str,
        before: &str,
        after: &str,
        sink: &crate::audit::AuditSink,
    ) -> crate::Result<&MutationRecord> {
        let seq = self.records.len() as u64;
        let prev_hash = self.head();
        let authorized = AUTHORIZED_SOURCES.contains(&source);
        let before_hash = sha_hex(before);
        let after_hash = sha_hex(after);
        let record_hash = compute_record_hash(
            seq, source, authorized, &before_hash, &after_hash, &prev_hash,
        );
        let rec = MutationRecord {
            seq,
            source: source.to_string(),
            authorized,
            before_hash,
            after_hash,
            prev_hash,
            record_hash,
        };
        // Persist first: if this fails the in-memory ledger does not advance.
        sink.append_mutation_json(serde_json::to_string(&rec)?)?;
        self.records.push(rec);
        Ok(self.records.last().unwrap())
    }

    pub fn records(&self) -> &[MutationRecord] {
        &self.records
    }

    /// Mutable access for tests that simulate tampering.
    #[cfg(test)]
    pub fn records_mut(&mut self) -> &mut [MutationRecord] {
        &mut self.records
    }

    /// The current chain head ("GENESIS" if empty).
    pub fn head(&self) -> String {
        self.records
            .last()
            .map(|r| r.record_hash.clone())
            .unwrap_or_else(|| "GENESIS".to_string())
    }

    /// Verify the hash chain is intact and unbroken.
    pub fn verify_chain(&self) -> bool {
        let mut prev = "GENESIS".to_string();
        for (i, r) in self.records.iter().enumerate() {
            if r.seq != i as u64 || r.prev_hash != prev {
                return false;
            }
            let expect = compute_record_hash(
                r.seq,
                &r.source,
                r.authorized,
                &r.before_hash,
                &r.after_hash,
                &r.prev_hash,
            );
            if expect != r.record_hash {
                return false;
            }
            prev = r.record_hash.clone();
        }
        true
    }

    /// Any unauthorized mutation present? (the provenance alarm)
    pub fn has_unauthorized(&self) -> bool {
        self.records
            .iter()
            .any(|r| !AUTHORIZED_SOURCES.contains(&r.source.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_verifies_and_head_is_genesis() {
        let log = ProvenanceLog::new();
        assert!(log.verify_chain());
        assert_eq!(log.head(), "GENESIS");
        assert!(!log.has_unauthorized());
    }

    #[test]
    fn authorized_mutation_is_not_flagged() {
        let mut log = ProvenanceLog::new();
        log.append("agent_self_evolution", "old miim", "new miim");
        assert!(log.verify_chain());
        assert!(!log.has_unauthorized());
        assert_eq!(log.records().len(), 1);
    }

    #[test]
    fn moral_filter_mutation_is_flagged_unauthorized() {
        let mut log = ProvenanceLog::new();
        log.append("moral_filter", "sanctioned miim", "tilted miim");
        assert!(log.verify_chain());
        assert!(log.has_unauthorized());
    }

    #[test]
    fn chain_links_records_in_order() {
        let mut log = ProvenanceLog::new();
        log.append("operator", "a", "b");
        log.append("moral_filter", "b", "c");
        let recs = log.records();
        assert_eq!(recs[0].prev_hash, "GENESIS");
        assert_eq!(recs[1].prev_hash, recs[0].record_hash);
        assert_eq!(log.head(), recs[1].record_hash);
    }

    #[test]
    fn tampering_breaks_chain_verification() {
        let mut log = ProvenanceLog::new();
        log.append("operator", "a", "b");
        log.append("operator", "b", "c");
        // Forge a record's after_hash without recomputing the chain.
        log.records_mut()[0].after_hash = "deadbeef".to_string();
        assert!(!log.verify_chain());
    }

    #[test]
    fn consistent_forgery_caught_by_next_prev_hash() {
        let mut log = ProvenanceLog::new();
        log.append("operator", "a", "b");
        log.append("operator", "b", "c");
        // Tamper record 0 AND recompute its record_hash consistently, so record 0
        // self-verifies — the forgery must still be caught by record 1's stale prev_hash.
        {
            let recs = log.records_mut();
            recs[0].after_hash = "de".repeat(32); // 64 hex chars
            let rh = compute_record_hash(
                recs[0].seq,
                &recs[0].source.clone(),
                recs[0].authorized,
                &recs[0].before_hash.clone(),
                &recs[0].after_hash.clone(),
                &recs[0].prev_hash.clone(),
            );
            recs[0].record_hash = rh;
        }
        assert!(!log.verify_chain());
    }

    #[test]
    fn teed_record_chains_identically_to_plain_append() {
        // Structural test for the persist-first ordering guarantee:
        // append_teed with a no-op Plain sink must produce a record with identical
        // chain properties to what plain append produces, and the chain must verify.
        let sink = crate::audit::AuditSink::disabled();
        let mut log = ProvenanceLog::new();
        // Prime with one record so the second has a real prev_hash.
        log.append("operator", "x", "y");

        // Capture what plain append would compute for the next seq.
        let expected_seq = log.records().len() as u64;
        let expected_prev = log.head();

        log.append_teed("moral_filter", "a", "b", &sink).unwrap();

        let rec = log.records().last().unwrap();
        assert_eq!(rec.seq, expected_seq, "seq must follow in-order");
        assert_eq!(rec.prev_hash, expected_prev, "prev_hash must be the pre-call head");
        assert_eq!(rec.source, "moral_filter");
        assert!(!rec.authorized);
        assert_eq!(log.head(), rec.record_hash, "head must equal the new record's hash");
        assert!(log.verify_chain(), "chain must be intact after teed append");
    }

    #[test]
    fn teed_mutation_lands_in_chained_log() {
        use crate::audit::chain::EntryKind;
        use crate::audit::store::{Mode, StoreCfg, TamperEvidentLog};
        let dir = tempfile::tempdir().unwrap();
        let tlog = TamperEvidentLog::open(StoreCfg {
            path: dir.path().join("audit.jsonl"),
            mode: Mode::Chained,
            signer: None,
            anchor: None,
            batch: 4,
            fail_open: false,
        })
        .unwrap();
        let sink = crate::audit::AuditSink::Chained(tlog);
        let mut log = ProvenanceLog::new();
        log.append_teed("moral_filter", "a", "b", &sink).unwrap();
        drop(sink);
        let text = std::fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
        let mutation_line = text.lines().nth(1).unwrap();
        let e = crate::audit::chain::parse_line(mutation_line).unwrap();
        assert_eq!(e.kind, EntryKind::Mutation);
        assert_eq!(e.body["source"], "moral_filter");
        assert_eq!(e.body["authorized"], false);
    }
}
