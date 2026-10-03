//! TamperEvidentLog: a single writer THREAD owns the chain head; callers
//! send entries over std::sync::mpsc and (write-ahead) block on a per-entry
//! ack until the line is durably written. Fail-closed: any writer error fails
//! the append (and therefore the gateway call) unless fail_open is set.
//!
//! Header (seq 0) records pubkey/mode/versions. In batched mode a checkpoint
//! entry carrying the batch Merkle-root signature is appended every `batch`
//! entries; every `anchor_every` entries an Anchor is published to the sink.

use super::anchor_sink::{Anchor, AnchorSink};
use super::chain::{build_line, EntryKind, GENESIS};
use super::merkle::merkle_root;
use super::sign::AuditSigner;
use crate::Result;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Plain,
    Chained,
    ChainedSigned,
    ChainedSignedBatched,
}

pub struct StoreCfg {
    pub path: PathBuf,
    pub mode: Mode,
    pub signer: Option<AuditSigner>,
    pub anchor: Option<(Box<dyn AnchorSink>, u64)>, // (sink, every-k)
    pub batch: u64,                                 // checkpoint cadence in batched mode
    pub fail_open: bool,
}

enum Msg {
    Append {
        kind: EntryKind,
        body_json: String,
        ack: mpsc::SyncSender<Result<u64>>,
    },
    Shutdown,
}

pub struct TamperEvidentLog {
    tx: mpsc::Sender<Msg>,
    fail_open: bool,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl TamperEvidentLog {
    pub fn open(cfg: StoreCfg) -> Result<Self> {
        let fail_open = cfg.fail_open;
        let (tx, rx) = mpsc::channel::<Msg>();
        let mut w = Writer::start(cfg)?;
        let handle = std::thread::spawn(move || w.run(rx));
        Ok(TamperEvidentLog {
            tx,
            fail_open,
            handle: Some(handle),
        })
    }

    /// Append one entry; blocks until durably written (write-ahead).
    /// Fail-closed: returns Err on any writer failure unless fail_open.
    pub fn append_json(&self, kind: EntryKind, body_json: String) -> Result<u64> {
        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        let sent = self.tx.send(Msg::Append {
            kind,
            body_json,
            ack: ack_tx,
        });
        let res = match sent {
            Ok(()) => match ack_rx.recv() {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!("audit writer died (recv)").into()),
            },
            Err(_) => Err(anyhow::anyhow!("audit writer died (send)").into()),
        };
        match res {
            Err(e) if self.fail_open => {
                tracing::warn!("audit append failed (fail_open): {e}");
                Ok(0)
            }
            other => other,
        }
    }
}

impl Drop for TamperEvidentLog {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct Writer {
    file: std::fs::File,
    mode: Mode,
    signer: Option<AuditSigner>,
    anchor: Option<(Box<dyn AnchorSink>, u64)>,
    batch: u64,
    next_seq: u64,
    prev_hash: String,
    /// this_hash of every entry so far (for anchors + batch roots).
    all_hashes: Vec<String>,
    /// hashes since the last checkpoint (batched mode).
    batch_hashes: Vec<String>,
}

impl Writer {
    fn start(cfg: StoreCfg) -> Result<Writer> {
        let exists = cfg.path.exists() && std::fs::metadata(&cfg.path)?.len() > 0;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&cfg.path)?;
        let mut w = Writer {
            file,
            mode: cfg.mode,
            signer: cfg.signer,
            anchor: cfg.anchor,
            batch: cfg.batch.max(1),
            next_seq: 0,
            prev_hash: GENESIS.to_string(),
            all_hashes: Vec::new(),
            batch_hashes: Vec::new(),
        };
        if exists {
            // Resume: recover head by scanning the existing chain.
            // Resume trusts the existing file's recorded hashes (no re-verification
            // here); a tampered head is caught by the verifier (verify.rs), and
            // entries appended after a tampered head chain from a hash the verifier
            // will reject.
            let text = std::fs::read_to_string(&cfg.path)?;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let e = super::chain::parse_line(line)?;
                w.next_seq = e.seq + 1;
                w.prev_hash = e.this_hash.clone();
                w.all_hashes.push(e.this_hash);
            }
        } else {
            // Fresh log: write the header entry (seq 0).
            let header = serde_json::json!({
                "format_version": super::chain::FORMAT_VERSION,
                "qfire_version": crate::VERSION,
                "mode": w.mode,
                "pubkey": w.signer.as_ref().map(|s| s.pubkey_hex()),
            });
            w.write_one(EntryKind::Header, header.to_string())?;
        }
        Ok(w)
    }

    fn write_one(&mut self, kind: EntryKind, body_json: String) -> Result<u64> {
        let ts = super::now_ts();
        let per_entry_sig = matches!(self.mode, Mode::ChainedSigned) && self.signer.is_some();
        let seq = self.next_seq;
        // Use the simple closure pattern to avoid borrow checker fights:
        // borrow self.signer and self.prev_hash independently, with a local
        // clone of prev_hash so there is no simultaneous mutable borrow conflict.
        let prev = self.prev_hash.clone();
        let (line, this_hash) = if per_entry_sig {
            let s = self.signer.as_ref().unwrap();
            let f = |h: &str| s.sign_hash_hex(h);
            build_line(seq, &ts, kind, &body_json, &prev, Some(&f))?
        } else {
            build_line(seq, &ts, kind, &body_json, &prev, None)?
        };
        writeln!(self.file, "{line}")?;
        self.file.sync_data()?;
        self.next_seq += 1;
        self.prev_hash = this_hash.clone();
        self.all_hashes.push(this_hash.clone());
        self.batch_hashes.push(this_hash);
        self.after_write(seq, kind)?;
        Ok(seq)
    }

    fn after_write(&mut self, seq: u64, kind: EntryKind) -> Result<()> {
        // Batched checkpoint: sign the Merkle root of the batch.
        // Guard: skip checkpoint logic for Checkpoint entries themselves to
        // prevent infinite recursion when batch==1.
        if matches!(self.mode, Mode::ChainedSignedBatched)
            && kind != EntryKind::Checkpoint
            && self.batch_hashes.len() as u64 >= self.batch
        {
            let root = merkle_root(&self.batch_hashes);
            let sig = self
                .signer
                .as_ref()
                .map(|s| s.sign_hash_hex(&root))
                .unwrap_or_default();
            let body = serde_json::json!({
                "checkpoint": "batch",
                "merkle_root": root,
                "sig": sig,
                "covers": self.batch_hashes.len()
            });
            // Clear BEFORE the recursive write_one — bounds the recursion:
            // the checkpoint entry itself starts the next batch (batch_hashes
            // will have 1 entry after write_one returns, not >= batch).
            self.batch_hashes.clear();
            // Recursion is bounded: checkpoint entries skip this branch (guard above).
            self.write_one(EntryKind::Checkpoint, body.to_string())?;
            // No early return: fall through so the anchor check runs on the
            // original entry's seq boundary as well.
        }
        // External anchor every k entries (runs for all entry kinds).
        if let Some((_, k)) = &self.anchor {
            let k = *k;
            if k > 0 && (seq + 1) % k == 0 {
                let root = merkle_root(&self.all_hashes);
                let sig = self
                    .signer
                    .as_ref()
                    .map(|s| s.sign_hash_hex(&root))
                    .unwrap_or_default();
                let a = Anchor {
                    seq,
                    merkle_root: root,
                    ts: super::now_ts(),
                    sig,
                };
                if let Some((sink, _)) = &mut self.anchor {
                    sink.publish(&a)?;
                }
            }
        }
        Ok(())
    }

    fn run(&mut self, rx: mpsc::Receiver<Msg>) {
        while let Ok(msg) = rx.recv() {
            match msg {
                Msg::Append {
                    kind,
                    body_json,
                    ack,
                } => {
                    let res = self.write_one(kind, body_json);
                    let _ = ack.send(res);
                }
                Msg::Shutdown => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::anchor_sink::FileAnchorSink;
    use crate::audit::chain::parse_line;
    use std::path::Path;
    use tempfile::tempdir;

    fn read_lines(p: &Path) -> Vec<String> {
        std::fs::read_to_string(p)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    fn open(dir: &Path, mode: Mode, signed: bool, anchor_k: Option<u64>) -> TamperEvidentLog {
        let signer = signed.then(|| AuditSigner::generate_to(&dir.join("key")).unwrap());
        let anchor = anchor_k.map(|k| {
            (
                Box::new(FileAnchorSink::new(dir.join("anchors.jsonl"))) as Box<dyn AnchorSink>,
                k,
            )
        });
        TamperEvidentLog::open(StoreCfg {
            path: dir.join("audit.jsonl"),
            mode,
            signer,
            anchor,
            batch: 4,
            fail_open: false,
        })
        .unwrap()
    }

    #[test]
    fn header_then_chained_entries() {
        let dir = tempdir().unwrap();
        let log = open(dir.path(), Mode::Chained, false, None);
        log.append_json(EntryKind::Decision, "{\"d\":1}".into())
            .unwrap();
        log.append_json(EntryKind::Decision, "{\"d\":2}".into())
            .unwrap();
        drop(log);
        let lines = read_lines(&dir.path().join("audit.jsonl"));
        assert_eq!(lines.len(), 3); // header + 2
        let e0 = parse_line(&lines[0]).unwrap();
        assert_eq!(e0.kind, EntryKind::Header);
        assert_eq!(e0.prev_hash, GENESIS);
        let e1 = parse_line(&lines[1]).unwrap();
        assert_eq!(e1.prev_hash, e0.this_hash);
        assert_eq!(parse_line(&lines[2]).unwrap().prev_hash, e1.this_hash);
    }

    #[test]
    fn signed_mode_signs_every_entry() {
        let dir = tempdir().unwrap();
        let log = open(dir.path(), Mode::ChainedSigned, true, None);
        log.append_json(EntryKind::Decision, "{}".into()).unwrap();
        drop(log);
        let lines = read_lines(&dir.path().join("audit.jsonl"));
        let e = parse_line(&lines[1]).unwrap();
        assert_eq!(e.sig.len(), 128);
    }

    #[test]
    fn batched_mode_emits_signed_checkpoints() {
        let dir = tempdir().unwrap();
        let log = open(dir.path(), Mode::ChainedSignedBatched, true, None);
        for i in 0..8 {
            log.append_json(EntryKind::Decision, format!("{{\"i\":{i}}}"))
                .unwrap();
        }
        drop(log);
        let lines = read_lines(&dir.path().join("audit.jsonl"));
        let checkpoints: Vec<_> = lines
            .iter()
            .filter(|l| parse_line(l).unwrap().kind == EntryKind::Checkpoint)
            .collect();
        // Trace (batch=4): header(h0),d0,d1,d2 → batch_hashes len=4 → cp1 emitted;
        // batch_hashes cleared; cp1's hash pushed (len=1); d3,d4,d5 → len=4 → cp2;
        // d6,d7 → len=3, no third checkpoint. Total checkpoints = 2.
        assert_eq!(
            checkpoints.len(),
            2,
            "expected 2 checkpoints in {} lines",
            lines.len()
        );
        let cp = parse_line(checkpoints[0]).unwrap();
        assert_eq!(cp.body["sig"].as_str().unwrap().len(), 128);
    }

    #[test]
    fn anchors_published_every_k() {
        let dir = tempdir().unwrap();
        let log = open(dir.path(), Mode::Chained, false, Some(5));
        for i in 0..12 {
            log.append_json(EntryKind::Decision, format!("{{\"i\":{i}}}"))
                .unwrap();
        }
        drop(log);
        let anchors = FileAnchorSink::read_all(dir.path().join("anchors.jsonl")).unwrap();
        assert!(anchors.len() >= 2, "got {} anchors", anchors.len());
    }

    #[test]
    fn resume_continues_the_chain() {
        let dir = tempdir().unwrap();
        {
            let log = open(dir.path(), Mode::Chained, false, None);
            log.append_json(EntryKind::Decision, "{\"a\":1}".into())
                .unwrap();
        }
        {
            let log = TamperEvidentLog::open(StoreCfg {
                path: dir.path().join("audit.jsonl"),
                mode: Mode::Chained,
                signer: None,
                anchor: None,
                batch: 4,
                fail_open: false,
            })
            .unwrap();
            log.append_json(EntryKind::Decision, "{\"a\":2}".into())
                .unwrap();
        }
        let lines = read_lines(&dir.path().join("audit.jsonl"));
        assert_eq!(lines.len(), 3);
        let prev = parse_line(&lines[1]).unwrap().this_hash;
        assert_eq!(parse_line(&lines[2]).unwrap().prev_hash, prev);
    }

    #[test]
    fn fail_open_swallows_writer_death_fail_closed_does_not() {
        // Simulate writer death by dropping the receiver: open a log, take
        // its thread down via Shutdown, then append.
        let dir = tempdir().unwrap();
        let log = open(dir.path(), Mode::Chained, false, None);
        log.tx.send(Msg::Shutdown).unwrap();
        // Give the thread a moment to exit.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(log.append_json(EntryKind::Decision, "{}".into()).is_err());
    }

    #[test]
    fn batch_of_one_terminates() {
        // batch=1 must not stack-overflow: each decision should trigger exactly
        // one checkpoint, and the checkpoint itself must NOT trigger another.
        let dir = tempdir().unwrap();
        let signer = AuditSigner::generate_to(&dir.path().join("key")).unwrap();
        let log = TamperEvidentLog::open(StoreCfg {
            path: dir.path().join("audit.jsonl"),
            mode: Mode::ChainedSignedBatched,
            signer: Some(signer),
            anchor: None,
            batch: 1,
            fail_open: false,
        })
        .unwrap();
        log.append_json(EntryKind::Decision, "{\"i\":0}".into())
            .unwrap();
        log.append_json(EntryKind::Decision, "{\"i\":1}".into())
            .unwrap();
        log.append_json(EntryKind::Decision, "{\"i\":2}".into())
            .unwrap();
        drop(log);

        let lines = read_lines(&dir.path().join("audit.jsonl"));
        let decisions: Vec<_> = lines
            .iter()
            .filter(|l| parse_line(l).unwrap().kind == EntryKind::Decision)
            .collect();
        let checkpoints: Vec<_> = lines
            .iter()
            .filter(|l| parse_line(l).unwrap().kind == EntryKind::Checkpoint)
            .collect();
        // Trace (batch=1): header → cp0; d0 → cp1; d1 → cp2; d2 → cp3.
        // The header is also a non-Checkpoint entry so it triggers one checkpoint too.
        // Total: 4 checkpoints for header + 3 decisions. No stack overflow = pass.
        assert_eq!(decisions.len(), 3);
        assert_eq!(
            checkpoints.len(),
            4,
            "header + 3 decisions each produce one checkpoint"
        );
    }

    #[test]
    fn fail_open_returns_ok_after_writer_death() {
        // Like the writer-death test but with fail_open:true; append must return Ok(0).
        let dir = tempdir().unwrap();
        let log = TamperEvidentLog::open(StoreCfg {
            path: dir.path().join("audit.jsonl"),
            mode: Mode::Chained,
            signer: None,
            anchor: None,
            batch: 4,
            fail_open: true,
        })
        .unwrap();
        log.tx.send(Msg::Shutdown).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(
            log.append_json(EntryKind::Decision, "{}".into()).unwrap(),
            0,
            "fail_open should return Ok(0) after writer death"
        );
    }
}
