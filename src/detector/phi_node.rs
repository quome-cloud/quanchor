//! HIPAA PHI detector node.
//!
//! Flags a prompt that carries Protected Health Information (any of the 18 HIPAA
//! Safe-Harbor identifiers). BLOCKs when the number of distinct PHI hits reaches
//! `min_hits`; otherwise ABSTAINs. The raw score is the PHI hit count, enabling
//! ROC/AUC analysis of PHI leakage detection.

use super::{elapsed_ms, now, DetectCtx, Detector};
use crate::phi;
use crate::verdict::{NodeVerdict, Verdict};
use async_trait::async_trait;

pub struct PhiDetector {
    min_hits: usize,
    config_hash: String,
}

impl PhiDetector {
    pub fn new(min_hits: usize, config_hash: String) -> Self {
        PhiDetector { min_hits: min_hits.max(1), config_hash }
    }
}

#[async_trait]
impl Detector for PhiDetector {
    fn kind(&self) -> &'static str {
        "phi"
    }
    fn version(&self) -> String {
        format!("hipaa-safe-harbor/{}ids", phi::covered_identifier_count())
    }
    fn config_hash(&self) -> &str {
        &self.config_hash
    }

    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict {
        let start = now();
        let hits = phi::detect(ctx.prompt);
        let n = hits.len();
        if n >= self.min_hits {
            let mut cats: Vec<&str> = hits.iter().map(|h| h.category).collect();
            cats.sort_unstable();
            cats.dedup();
            NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Block,
                (0.6 + 0.1 * n as f64).min(0.99),
                elapsed_ms(start),
                format!("{n} PHI element(s): {}", cats.join(", ")),
            )
            .with_score(n as f64)
        } else {
            NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Abstain,
                0.5,
                elapsed_ms(start),
                "no PHI detected",
            )
            .with_score(0.0)
        }
    }
}
