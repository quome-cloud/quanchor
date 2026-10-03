//! consent::ledger — G5 participatory-governance ledger. An append-only,
//! hash-chained record of the participatory-governance inputs that legitimise a
//! patient-facing agent's deployment (C4.1): community-review events, patient-
//! advisory sign-offs, public-comment summaries. Each entry chains to the
//! previous one exactly as the 003 audit log does, so any retroactive edit breaks
//! verification; the head hash is meant to be anchored into the 003 chain.
//!
//! There is no hypothesis test for G5 — this is a mechanism with a tamper-evidence
//! guarantee, demonstrated by the tests below.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One participatory-governance input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GovernanceEntry {
    /// e.g. "community_review", "patient_advisory_signoff", "public_comment".
    pub kind: String,
    /// Who provided the input (committee, advocate, board).
    pub party: String,
    /// Free-text summary of the input / decision.
    pub summary: String,
    /// Hex SHA-256 of the previous entry's `hash` ("" for the genesis entry).
    pub prev_hash: String,
    /// Hex SHA-256 over `prev_hash | kind | party | summary`.
    pub hash: String,
}

fn entry_hash(prev_hash: &str, kind: &str, party: &str, summary: &str) -> String {
    let mut h = Sha256::new();
    h.update(prev_hash.as_bytes());
    h.update(b"|");
    h.update(kind.as_bytes());
    h.update(b"|");
    h.update(party.as_bytes());
    h.update(b"|");
    h.update(summary.as_bytes());
    hex::encode(h.finalize())
}

/// An append-only, hash-chained ledger of governance inputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GovernanceLedger {
    pub entries: Vec<GovernanceEntry>,
}

impl GovernanceLedger {
    pub fn new() -> Self {
        GovernanceLedger::default()
    }

    /// Append an input, chaining it to the current head.
    pub fn append(&mut self, kind: &str, party: &str, summary: &str) -> &GovernanceEntry {
        let prev_hash = self.head().to_string();
        let hash = entry_hash(&prev_hash, kind, party, summary);
        self.entries.push(GovernanceEntry {
            kind: kind.to_string(),
            party: party.to_string(),
            summary: summary.to_string(),
            prev_hash,
            hash,
        });
        self.entries.last().unwrap()
    }

    /// The head hash to anchor into the 003 chain ("" when empty).
    pub fn head(&self) -> &str {
        self.entries.last().map(|e| e.hash.as_str()).unwrap_or("")
    }

    /// Verify the whole chain: every entry's hash recomputes and links to its
    /// predecessor. Returns the index of the first broken entry, or `None` if the
    /// ledger is intact.
    pub fn verify(&self) -> Option<usize> {
        let mut prev = String::new();
        for (i, e) in self.entries.iter().enumerate() {
            if e.prev_hash != prev {
                return Some(i);
            }
            let expect = entry_hash(&e.prev_hash, &e.kind, &e.party, &e.summary);
            if e.hash != expect {
                return Some(i);
            }
            prev = e.hash.clone();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> GovernanceLedger {
        let mut l = GovernanceLedger::new();
        l.append("community_review", "East-Side Patient Council", "Reviewed reminder agent; approved with opt-out requirement.");
        l.append("patient_advisory_signoff", "Advisory Board", "Signed off on disclosure language.");
        l.append("public_comment", "Public", "12 comments; no objections after opt-out added.");
        l
    }

    #[test]
    fn intact_chain_verifies() {
        assert_eq!(ledger().verify(), None);
    }

    #[test]
    fn head_advances_on_append() {
        let mut l = GovernanceLedger::new();
        assert_eq!(l.head(), "");
        l.append("community_review", "X", "first");
        let h1 = l.head().to_string();
        l.append("public_comment", "Y", "second");
        assert_ne!(l.head(), h1);
        assert_eq!(l.entries[1].prev_hash, h1);
    }

    #[test]
    fn tampering_summary_breaks_verification() {
        let mut l = ledger();
        l.entries[1].summary = "Signed off on WEAKER disclosure language.".into();
        assert_eq!(l.verify(), Some(1));
    }

    #[test]
    fn deleting_an_entry_breaks_the_link() {
        let mut l = ledger();
        l.entries.remove(1);
        // Entry that was at index 2 now sits at 1 with a stale prev_hash.
        assert_eq!(l.verify(), Some(1));
    }

    #[test]
    fn reordering_breaks_verification() {
        let mut l = ledger();
        l.entries.swap(0, 1);
        assert_eq!(l.verify(), Some(0));
    }
}
