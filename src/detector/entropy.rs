//! Shannon-entropy and length heuristics for obfuscation / payload smuggling.

use super::{elapsed_ms, now, DetectCtx, Detector};
use crate::verdict::{NodeVerdict, Verdict};
use async_trait::async_trait;

pub struct EntropyDetector {
    max_bits: f64,
    max_len: Option<usize>,
    max_blob: usize,
    config_hash: String,
}

impl EntropyDetector {
    pub fn new(
        max_bits: Option<f64>,
        max_len: Option<usize>,
        max_blob: Option<usize>,
        config_hash: String,
    ) -> Self {
        EntropyDetector {
            max_bits: max_bits.unwrap_or(4.5),
            max_len,
            max_blob: max_blob.unwrap_or(40),
            config_hash,
        }
    }
}

/// Shannon entropy (bits per character) of a string.
pub fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut counts = std::collections::HashMap::new();
    let mut n = 0usize;
    for c in s.chars() {
        *counts.entry(c).or_insert(0usize) += 1;
        n += 1;
    }
    let n = n as f64;
    counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// The length of the longest run of base64/hex-like characters.
pub fn longest_blob(s: &str) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=' {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[async_trait]
impl Detector for EntropyDetector {
    fn kind(&self) -> &'static str {
        "entropy"
    }
    fn version(&self) -> String {
        format!("shannon/maxbits={:.2}", self.max_bits)
    }
    fn config_hash(&self) -> &str {
        &self.config_hash
    }

    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict {
        let start = now();
        let bits = shannon_entropy(ctx.prompt);
        let blob = longest_blob(ctx.prompt);
        let len = ctx.prompt.chars().count();

        if let Some(max_len) = self.max_len {
            if len > max_len {
                return NodeVerdict::new(
                    self.kind(),
                    self.version(),
                    Verdict::Block,
                    0.7,
                    elapsed_ms(start),
                    format!("prompt length {len} exceeds max {max_len}"),
                )
                .with_score(bits);
            }
        }
        if blob >= self.max_blob {
            return NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Block,
                0.65,
                elapsed_ms(start),
                format!("encoded blob of length {blob} (possible payload smuggling)"),
            )
            .with_score(bits);
        }
        if bits > self.max_bits {
            return NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Block,
                ((bits - self.max_bits) / 2.0).clamp(0.0, 1.0),
                elapsed_ms(start),
                format!("entropy {bits:.2} bits/char exceeds max {:.2}", self.max_bits),
            )
            .with_score(bits);
        }
        NodeVerdict::new(
            self.kind(),
            self.version(),
            Verdict::Abstain,
            0.5,
            elapsed_ms(start),
            format!("entropy {bits:.2} bits/char within bounds"),
        )
        .with_score(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_of_uniform_is_low() {
        assert!(shannon_entropy("aaaaaaaa") < 0.5);
    }

    #[test]
    fn entropy_of_random_is_high() {
        let s = "aZ9bQ2xK7mP1wR4tY6uI0oL5";
        assert!(shannon_entropy(s) > 3.5);
    }

    #[test]
    fn longest_blob_detects_base64_run() {
        assert!(longest_blob("hello aGVsbG8gd29ybGQK world") >= 12);
        assert!(longest_blob("hi there friend") < 6);
    }
}
