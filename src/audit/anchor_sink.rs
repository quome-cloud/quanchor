//! External anchoring (TA6 rollback defense). Every k entries the store
//! publishes {seq, merkle_root, ts, sig} to an AnchorSink. FileAnchorSink
//! appends JSONL to a separate path (simulating a separate host). A
//! GCS-object-lock sink is wired at enclave-deploy time (out of scope here).

use crate::Result;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anchor {
    /// Sequence number of the last entry covered by this anchor.
    pub seq: u64,
    /// Merkle root over this_hash of ALL entries 0..=seq.
    pub merkle_root: String,
    pub ts: String,
    /// ed25519 over `merkle_root` hex bytes ("" when unsigned).
    pub sig: String,
}

pub trait AnchorSink: Send {
    fn publish(&mut self, anchor: &Anchor) -> Result<()>;
}

/// Appends anchors as JSONL to a file on a separate path.
pub struct FileAnchorSink {
    path: PathBuf,
}

impl FileAnchorSink {
    pub fn new(path: impl AsRef<Path>) -> Self {
        FileAnchorSink {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Read all anchors back (verifier side).
    pub fn read_all(path: impl AsRef<Path>) -> Result<Vec<Anchor>> {
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            out.push(serde_json::from_str(line)?);
        }
        Ok(out)
    }
}

impl AnchorSink for FileAnchorSink {
    fn publish(&mut self, anchor: &Anchor) -> Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{}", serde_json::to_string(anchor)?)?;
        f.sync_data()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn publish_then_read_roundtrips_in_order() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("anchors.jsonl");
        let mut sink = FileAnchorSink::new(&p);
        for i in 0..3u64 {
            sink.publish(&Anchor {
                seq: i * 100,
                merkle_root: format!("{i:064}"),
                ts: "t".into(),
                sig: String::new(),
            })
            .unwrap();
        }
        let back = FileAnchorSink::read_all(&p).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[2].seq, 200);
    }
}
