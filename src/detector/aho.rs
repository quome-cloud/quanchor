//! Aho-Corasick multi-pattern denylist detector.
//!
//! Compiles a (potentially large) keyword denylist into a single Aho-Corasick
//! automaton for O(n) scanning regardless of list size — far cheaper than many
//! independent regexes when a rule carries hundreds of literal markers.

use super::{elapsed_ms, now, DetectCtx, Detector};
use crate::verdict::{NodeVerdict, Verdict};
use crate::Result;
use aho_corasick::{AhoCorasick, MatchKind};
use async_trait::async_trait;

pub struct AhoDetector {
    ac: AhoCorasick,
    patterns: Vec<String>,
    config_hash: String,
}

impl AhoDetector {
    pub fn new(deny: &[String], config_hash: String) -> Result<Self> {
        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(deny)
            .map_err(|e| crate::Error::Config(format!("aho-corasick build: {e}")))?;
        Ok(AhoDetector { ac, patterns: deny.to_vec(), config_hash })
    }
}

#[async_trait]
impl Detector for AhoDetector {
    fn kind(&self) -> &'static str {
        "aho"
    }
    fn version(&self) -> String {
        format!("aho-corasick/{}terms", self.patterns.len())
    }
    fn config_hash(&self) -> &str {
        &self.config_hash
    }

    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict {
        let start = now();
        let mut hits = Vec::new();
        for m in self.ac.find_iter(ctx.prompt) {
            let term = self.patterns[m.pattern().as_u32() as usize].clone();
            if !hits.contains(&term) {
                hits.push(term);
            }
            if hits.len() >= 3 {
                break;
            }
        }
        if !hits.is_empty() {
            NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Block,
                0.9,
                elapsed_ms(start),
                format!("matched denylist term(s): {}", hits.join(", ")),
            )
            .with_score(1.0)
        } else {
            NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Abstain,
                0.5,
                elapsed_ms(start),
                "no denylist term matched",
            )
            .with_score(0.0)
        }
    }
}
