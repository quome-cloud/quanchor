//! Regex / keyword / denylist matcher.

use super::{elapsed_ms, now, DetectCtx, Detector};
use crate::verdict::{NodeVerdict, Verdict};
use crate::Result;
use async_trait::async_trait;
use regex::RegexSet;

pub struct RegexDetector {
    deny: RegexSet,
    deny_src: Vec<String>,
    allow: RegexSet,
    config_hash: String,
}

impl RegexDetector {
    pub fn new(
        deny: &[String],
        allow: &[String],
        keywords: &[String],
        config_hash: String,
    ) -> Result<Self> {
        // Keywords are compiled as case-insensitive escaped substrings folded
        // into the deny set.
        let mut deny_patterns: Vec<String> = deny.to_vec();
        for kw in keywords {
            deny_patterns.push(format!("(?i){}", regex::escape(kw)));
        }
        let deny_set = RegexSet::new(&deny_patterns)?;
        let allow_set = RegexSet::new(allow)?;
        Ok(RegexDetector {
            deny: deny_set,
            deny_src: deny_patterns,
            allow: allow_set,
            config_hash,
        })
    }
}

#[async_trait]
impl Detector for RegexDetector {
    fn kind(&self) -> &'static str {
        "regex"
    }
    fn version(&self) -> String {
        format!("regex-set/{}deny", self.deny.len())
    }
    fn config_hash(&self) -> &str {
        &self.config_hash
    }

    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict {
        let start = now();
        // Allowlist wins: an explicit allow match short-circuits to ALLOW.
        if !self.allow.is_empty() && self.allow.is_match(ctx.prompt) {
            return NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Allow,
                0.9,
                elapsed_ms(start),
                "matched an allowlist pattern",
            );
        }
        let matches: Vec<usize> = self.deny.matches(ctx.prompt).into_iter().collect();
        if !matches.is_empty() {
            let which = matches
                .iter()
                .take(2)
                .map(|&i| self.deny_src.get(i).cloned().unwrap_or_default())
                .collect::<Vec<_>>()
                .join(", ");
            return NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Block,
                0.95,
                elapsed_ms(start),
                format!("matched denylist pattern(s): {which}"),
            )
            .with_score(1.0);
        }
        // No denylist hit: abstain (let downstream nodes decide scope).
        NodeVerdict::new(
            self.kind(),
            self.version(),
            Verdict::Abstain,
            0.5,
            elapsed_ms(start),
            "no denylist match",
        )
        .with_score(0.0)
    }
}
