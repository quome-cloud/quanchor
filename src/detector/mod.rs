//! The detector library.
//!
//! Every detector node implements the [`Detector`] trait and returns a
//! [`NodeVerdict`] (verdict + confidence + latency + rationale + optional raw
//! score). Nodes are declared in YAML via [`NodeConfig`] and built by
//! [`build_detector`]. Each detector reports a `kind` and a `version` so results
//! are reproducible and citable.

mod aho;
mod custom;
mod deberta;
mod entropy;
mod goal_drift;
mod judge;
mod output_monitor;
mod phi_node;
mod regex_node;
mod similarity;

pub use aho::AhoDetector;
pub use custom::CustomDetector;
pub use deberta::{DebertaDetector, lexical_injection_score};
pub use entropy::EntropyDetector;
pub use goal_drift::GoalDriftDetector;
pub use judge::JudgeDetector;
pub use output_monitor::OutputMonitorDetector;
pub use phi_node::PhiDetector;
pub use regex_node::RegexDetector;
pub use similarity::SimilarityDetector;

use crate::provider::ProviderRegistry;
use crate::rule::Exemplars;
use crate::verdict::NodeVerdict;
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The evaluation context handed to every detector node.
pub struct DetectCtx<'a> {
    /// The flattened prompt text under evaluation.
    pub prompt: &'a str,
    /// The rule's natural-language scope description (the sanctioned MIIM).
    pub scope: &'a str,
    /// The rule's in-scope / out-of-scope exemplars.
    pub exemplars: &'a Exemplars,
    /// Provider registry, for detectors that make LLM calls (the judge node).
    pub providers: &'a ProviderRegistry,
    /// The effective scope after any harness rewrite (the live, possibly-tilted
    /// MIIM). `None` ⇒ no harness rewrite happened; detectors fall back to `scope`.
    pub effective_scope: Option<&'a str>,
    /// The downstream response text, for post-response output monitors. `None`
    /// during the pre-forward firewall pass.
    pub response: Option<&'a str>,
}

impl<'a> DetectCtx<'a> {
    /// Construct a context for the pre-response firewall pass.
    pub fn new(
        prompt: &'a str,
        scope: &'a str,
        exemplars: &'a Exemplars,
        providers: &'a ProviderRegistry,
    ) -> Self {
        DetectCtx { prompt, scope, exemplars, providers, effective_scope: None, response: None }
    }

    /// Attach the effective (post-harness) scope.
    pub fn with_effective_scope(mut self, eff: &'a str) -> Self {
        self.effective_scope = Some(eff);
        self
    }

    /// Attach the downstream response text.
    pub fn with_response(mut self, response: &'a str) -> Self {
        self.response = Some(response);
        self
    }
}

/// A detector node: the unit of evaluation inside a rule's pipeline.
#[async_trait]
pub trait Detector: Send + Sync {
    /// The detector kind, e.g. `"regex"`.
    fn kind(&self) -> &'static str;
    /// A model/version string for reproducibility.
    fn version(&self) -> String;
    /// A stable hash of this node's configuration, used in the verdict cache key.
    fn config_hash(&self) -> &str;
    /// Whether this detector performs network I/O (used to order cheap-before-
    /// expensive and to attribute firewall cost). Default: false.
    fn is_expensive(&self) -> bool {
        false
    }
    /// Evaluate the prompt and return a verdict.
    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict;
}

/// The declarative configuration of one detector node, as written in a rule's
/// YAML `pipeline`. The `type` field selects the variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeConfig {
    /// Regex / keyword / denylist matcher.
    Regex {
        /// Patterns that, if matched, produce BLOCK.
        #[serde(default)]
        deny: Vec<String>,
        /// Patterns that, if matched, produce ALLOW (allowlist short-circuit).
        #[serde(default)]
        allow: Vec<String>,
        /// Plain (case-insensitive) keywords treated as deny substrings.
        #[serde(default)]
        keywords: Vec<String>,
    },
    /// Shannon-entropy and length heuristics for obfuscation / payload smuggling.
    Entropy {
        /// BLOCK when normalized entropy (bits/char) exceeds this. Default 4.5.
        #[serde(default)]
        max_bits: Option<f64>,
        /// BLOCK when the prompt is longer than this many chars. Optional.
        #[serde(default)]
        max_len: Option<usize>,
        /// BLOCK long base64/hex blobs of at least this length. Default 40.
        #[serde(default)]
        max_blob: Option<usize>,
    },
    /// Local prompt-injection classifier (protectai/deberta-v3 via ONNX, or a
    /// lexical fallback when the `onnx` feature is not compiled in).
    Deberta {
        /// BLOCK when injection probability exceeds this. Default 0.5.
        #[serde(default)]
        threshold: Option<f64>,
    },
    /// LLM scope-judge: a single classification call against a provider.
    Judge {
        /// Provider profile name; defaults to the registry default.
        #[serde(default)]
        provider: Option<String>,
        /// Model override; defaults to the profile's model.
        #[serde(default)]
        model: Option<String>,
    },
    /// Similarity of the prompt to the rule's in-scope exemplars.
    Similarity {
        /// BLOCK when max similarity is below this. Default 0.15.
        #[serde(default)]
        threshold: Option<f64>,
        /// Optional Ollama embedding model for true semantic embeddings.
        #[serde(default)]
        embed_model: Option<String>,
        /// Provider profile for embeddings (Ollama). Defaults to registry default.
        #[serde(default)]
        provider: Option<String>,
    },
    /// A user-supplied check: an external script or an embedded keyword rule.
    Custom {
        /// Path to an executable that reads JSON on stdin and writes a verdict.
        #[serde(default)]
        command: Option<String>,
    },
    /// Aho-Corasick multi-pattern literal denylist (fast for large lists).
    Aho {
        #[serde(default)]
        deny: Vec<String>,
    },
    /// HIPAA PHI detector: BLOCK when at least `min_hits` Safe-Harbor identifiers
    /// are present.
    Phi {
        #[serde(default)]
        min_hits: Option<usize>,
    },
    /// Goal-drift: BLOCK when the live intent (effective scope) diverges from a
    /// sealed goal anchor — the harness goal-integrity defense.
    GoalDrift {
        /// The sanctioned MIIM (master instruction / intent) to anchor against.
        anchor: String,
        /// Seal key. In production it is released into the enclave via KMS/attestation;
        /// for local/dev it defaults from the `QFIRE_ANCHOR_KEY` env var. The built-in
        /// literal fallback is insecure (public) and provides no tamper protection.
        #[serde(default)]
        key: Option<String>,
        /// BLOCK when drift exceeds this. Default 0.3.
        #[serde(default)]
        threshold: Option<f64>,
        /// Optional Ollama embed model for semantic drift (falls back to lexical).
        #[serde(default)]
        embed_model: Option<String>,
        /// Provider profile for embeddings. Defaults to the registry default.
        #[serde(default)]
        provider: Option<String>,
    },
    /// Behavioral output monitor: BLOCK a response with a value outside `safe_range`
    /// or containing a `deny_terms` substring. Abstains pre-response.
    OutputMonitor {
        /// Inclusive [min, max]; values outside ⇒ BLOCK.
        #[serde(default)]
        safe_range: Option<(f64, f64)>,
        /// Substrings whose presence ⇒ BLOCK.
        #[serde(default)]
        deny_terms: Vec<String>,
    },
}

/// A stable JSON-based hash of a node config, for the verdict cache.
pub fn config_hash(node: &NodeConfig) -> String {
    let json = serde_json::to_string(node).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    hex::encode(&hasher.finalize()[..8])
}

/// Build a boxed detector from its declarative config.
pub fn build_detector(node: &NodeConfig) -> Result<Box<dyn Detector>> {
    let h = config_hash(node);
    let d: Box<dyn Detector> = match node {
        NodeConfig::Regex { deny, allow, keywords } => {
            Box::new(RegexDetector::new(deny, allow, keywords, h)?)
        }
        NodeConfig::Entropy { max_bits, max_len, max_blob } => {
            Box::new(EntropyDetector::new(*max_bits, *max_len, *max_blob, h))
        }
        NodeConfig::Deberta { threshold } => {
            Box::new(DebertaDetector::new(threshold.unwrap_or(0.5), h))
        }
        NodeConfig::Judge { provider, model } => {
            Box::new(JudgeDetector::new(provider.clone(), model.clone(), h))
        }
        NodeConfig::Similarity { threshold, embed_model, provider } => Box::new(
            SimilarityDetector::new(threshold.unwrap_or(0.15), embed_model.clone(), provider.clone(), h),
        ),
        NodeConfig::Custom { command } => {
            let cmd = command
                .clone()
                .ok_or_else(|| Error::Config("custom node requires a `command`".into()))?;
            Box::new(CustomDetector::new(cmd, h))
        }
        NodeConfig::Aho { deny } => Box::new(AhoDetector::new(deny, h)?),
        NodeConfig::Phi { min_hits } => Box::new(PhiDetector::new(min_hits.unwrap_or(1), h)),
        NodeConfig::GoalDrift { anchor, key, threshold, embed_model, provider } => {
            Box::new(GoalDriftDetector::new(
                anchor.clone(),
                key.clone().unwrap_or_else(|| {
                    // In production the key is released into the enclave via KMS/attestation.
                    // For local/dev use, read QFIRE_ANCHOR_KEY; the literal fallback is INSECURE
                    // (publicly known) and must not be relied on for tamper protection.
                    std::env::var("QFIRE_ANCHOR_KEY")
                        .unwrap_or_else(|_| "quokkafire-default-anchor-key".to_string())
                }),
                threshold.unwrap_or(0.3),
                h,
            ).with_embeddings(embed_model.clone(), provider.clone()))
        }
        NodeConfig::OutputMonitor { safe_range, deny_terms } => {
            Box::new(OutputMonitorDetector::new(*safe_range, deny_terms.clone(), h))
        }
    };
    Ok(d)
}

/// Convenience: a millisecond-resolution wall clock for detector latency.
pub(crate) fn now() -> std::time::Instant {
    std::time::Instant::now()
}

pub(crate) fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod ctx_tests {
    use super::*;
    use crate::config::Config;
    use crate::provider::ProviderRegistry;
    use crate::rule::Exemplars;

    #[test]
    fn detectctx_new_defaults_optionals_to_none() {
        let ex = Exemplars::default();
        let providers = ProviderRegistry::from_profiles(&Config::default().providers).unwrap();
        let ctx = DetectCtx::new("p", "s", &ex, &providers);
        assert!(ctx.effective_scope.is_none());
        assert!(ctx.response.is_none());
        let ctx2 = ctx.with_effective_scope("eff").with_response("resp");
        assert_eq!(ctx2.effective_scope, Some("eff"));
        assert_eq!(ctx2.response, Some("resp"));
        assert_eq!(ctx2.prompt, "p");
        assert_eq!(ctx2.scope, "s");
    }
}
