//! The immutable, attributable audit log.
//!
//! Every proxy decision and every benchmark trial is appended as one JSON line
//! (JSONL): timestamp, prompt hash, chain + rule + detector versions, per-node
//! verdicts, terminal decision, provider, model, tokens, cost and latency. The
//! log is append-only and is the system of record for live monitoring and
//! offline reproducibility.

pub mod anchor_sink;
pub mod chain;
pub mod merkle;
pub mod sign;
pub mod store;
pub mod verify;

use crate::engine::Decision;
use crate::ir::Usage;
use crate::verdict::Verdict;
use crate::Result;
use chrono::Utc;

/// Entry/anchor timestamp. Honors `QFIRE_AUDIT_FIXED_TS` for reproducible fixtures and tests
/// (the TamperBench E1 fixture must be byte-identical across runs, otherwise tamper *localization*
/// is non-deterministic even under a fixed tamper seed); otherwise the wall clock.
pub(crate) fn now_ts() -> String {
    std::env::var("QFIRE_AUDIT_FIXED_TS").unwrap_or_else(|_| Utc::now().to_rfc3339())
}
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One audit record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub ts: String,
    pub qfire_version: String,
    pub event: String,
    pub prompt_hash: String,
    pub chain_id: String,
    pub chain_version: String,
    pub terminal: Verdict,
    pub deciding_rule: Option<String>,
    pub deciding_node: Option<String>,
    pub reason: String,
    pub wall_clock_ms: f64,
    pub summed_detector_ms: f64,
    /// Compact per-node summary: rule_id, node kind, version, verdict, confidence.
    pub nodes: Vec<NodeSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A compact node line in an audit record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummary {
    pub rule: String,
    pub kind: String,
    pub version: String,
    pub verdict: Verdict,
    pub confidence: f64,
    pub latency_ms: f64,
}

impl AuditRecord {
    /// Build an audit record from a decision.
    pub fn from_decision(event: &str, decision: &Decision) -> Self {
        let mut nodes = Vec::new();
        for rt in &decision.trace.rules {
            for n in &rt.nodes {
                nodes.push(NodeSummary {
                    rule: rt.rule_id.clone(),
                    kind: n.kind.clone(),
                    version: n.version.clone(),
                    verdict: n.verdict,
                    confidence: n.confidence,
                    latency_ms: n.latency_ms,
                });
            }
        }
        AuditRecord {
            ts: now_ts(),
            qfire_version: crate::VERSION.to_string(),
            event: event.to_string(),
            prompt_hash: decision.prompt_hash.clone(),
            chain_id: decision.trace.chain_id.clone(),
            chain_version: decision.trace.chain_version.clone(),
            terminal: decision.terminal,
            deciding_rule: decision.deciding_rule.clone(),
            deciding_node: decision.deciding_node.clone(),
            reason: decision.reason.clone(),
            wall_clock_ms: decision.trace.wall_clock_ms,
            summed_detector_ms: decision.trace.summed_detector_ms,
            nodes,
            provider: None,
            model: None,
            usage: None,
        }
    }

    pub fn with_downstream(mut self, provider: &str, model: &str, usage: Usage) -> Self {
        self.provider = Some(provider.to_string());
        self.model = Some(model.to_string());
        self.usage = Some(usage);
        self
    }
}

/// Either the v1 plain JSONL log or the 003 tamper-evident chained log.
/// Same `append` surface so `App`/proxy/CLI call sites do not change shape.
pub enum AuditSink {
    Plain(AuditLog),
    Chained(store::TamperEvidentLog),
}

impl AuditSink {
    pub fn append(&self, record: &AuditRecord) -> crate::Result<()> {
        match self {
            AuditSink::Plain(l) => l.append(record),
            AuditSink::Chained(l) => {
                let body = serde_json::to_string(record)?;
                l.append_json(chain::EntryKind::Decision, body)?;
                Ok(())
            }
        }
    }

    /// A harness MIIM mutation teed into the same chain (no-op for Plain).
    pub fn append_mutation_json(&self, body_json: String) -> crate::Result<()> {
        match self {
            AuditSink::Plain(_) => Ok(()),
            AuditSink::Chained(l) => {
                l.append_json(chain::EntryKind::Mutation, body_json)?;
                Ok(())
            }
        }
    }

    /// A supply-chain admission record stamped into the chain (paper 004).
    /// No-op for the plain sink.
    pub fn append_admission_json(&self, body_json: String) -> crate::Result<()> {
        match self {
            AuditSink::Plain(_) => Ok(()),
            AuditSink::Chained(l) => {
                l.append_json(chain::EntryKind::Admission, body_json)?;
                Ok(())
            }
        }
    }

    /// A policy authorization decision stamped into the chain (paper 005).
    /// No-op for the plain sink.
    pub fn append_authorization_json(&self, body_json: String) -> crate::Result<()> {
        match self {
            AuditSink::Plain(_) => Ok(()),
            AuditSink::Chained(l) => {
                l.append_json(chain::EntryKind::Authorization, body_json)?;
                Ok(())
            }
        }
    }

    /// A retrieval / memory event stamped into the chain (paper 007).
    pub fn append_retrieval_json(&self, body_json: String) -> crate::Result<()> {
        match self {
            AuditSink::Plain(_) => Ok(()),
            AuditSink::Chained(l) => {
                l.append_json(chain::EntryKind::Retrieval, body_json)?;
                Ok(())
            }
        }
    }

    pub fn disabled() -> Self {
        AuditSink::Plain(AuditLog::disabled())
    }
}

/// An append-only audit log writer.
pub struct AuditLog {
    path: PathBuf,
    lock: Mutex<()>,
    enabled: bool,
}

impl AuditLog {
    /// Open (or create) an audit log at `path`.
    pub fn open(path: impl AsRef<Path>) -> Self {
        AuditLog {
            path: path.as_ref().to_path_buf(),
            lock: Mutex::new(()),
            enabled: true,
        }
    }

    /// A no-op audit log (used when auditing is disabled, e.g. dry-run explain).
    pub fn disabled() -> Self {
        AuditLog {
            path: PathBuf::new(),
            lock: Mutex::new(()),
            enabled: false,
        }
    }

    /// Append a record as one JSON line. Thread-safe.
    pub fn append(&self, record: &AuditRecord) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let line = serde_json::to_string(record)?;
        let _guard = self.lock.lock().unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Read all records back (for `qfire report` / offline analysis).
    pub fn read_all(path: impl AsRef<Path>) -> Result<Vec<AuditRecord>> {
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<AuditRecord>(line) {
                out.push(rec);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_entry_roundtrips_through_chain() {
        use crate::audit::chain::{parse_line, EntryKind};
        use crate::audit::store::{Mode, StoreCfg, TamperEvidentLog};
        let dir = tempfile::tempdir().unwrap();
        let log = TamperEvidentLog::open(StoreCfg {
            path: dir.path().join("audit.jsonl"),
            mode: Mode::Chained,
            signer: None,
            anchor: None,
            batch: 4,
            fail_open: false,
        })
        .unwrap();
        let sink = AuditSink::Chained(log);
        sink.append_admission_json("{\"aibom_digest\":\"abc\",\"components\":6}".into())
            .unwrap();
        drop(sink);
        let text = std::fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
        let line = text.lines().nth(1).unwrap(); // line 0 = header
        let e = parse_line(line).unwrap();
        assert_eq!(e.kind, EntryKind::Admission);
        assert_eq!(e.body["components"], 6);
    }

    #[test]
    fn authorization_entry_roundtrips_through_chain() {
        use crate::audit::chain::{parse_line, EntryKind};
        use crate::audit::store::{Mode, StoreCfg, TamperEvidentLog};
        let dir = tempfile::tempdir().unwrap();
        let log = TamperEvidentLog::open(StoreCfg {
            path: dir.path().join("audit.jsonl"), mode: Mode::Chained, signer: None,
            anchor: None, batch: 4, fail_open: false }).unwrap();
        let sink = AuditSink::Chained(log);
        sink.append_authorization_json("{\"effect\":\"deny\",\"policy_version\":\"v1\"}".into()).unwrap();
        drop(sink);
        let text = std::fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
        let e = parse_line(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(e.kind, EntryKind::Authorization);
        assert_eq!(e.body["effect"], "deny");
    }

    #[test]
    fn retrieval_entry_roundtrips_through_chain() {
        use crate::audit::chain::{parse_line, EntryKind};
        use crate::audit::store::{Mode, StoreCfg, TamperEvidentLog};
        let dir = tempfile::tempdir().unwrap();
        let log = TamperEvidentLog::open(StoreCfg {
            path: dir.path().join("audit.jsonl"), mode: Mode::Chained, signer: None,
            anchor: None, batch: 4, fail_open: false }).unwrap();
        let sink = AuditSink::Chained(log);
        sink.append_retrieval_json("{\"event\":\"retrieve\",\"k\":3}".into()).unwrap();
        drop(sink);
        let text = std::fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
        let e = parse_line(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(e.kind, EntryKind::Retrieval);
        assert_eq!(e.body["k"], 3);
    }
}
