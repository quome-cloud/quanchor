//! Paper 007 — memory & RAG-poisoning resistance. Documents and memory entries
//! carry a provenance trust tier (and an ed25519 signature when from a trusted
//! source); retrieval scores by trust×relevance over a TurboQuant quantized index,
//! flags poison (instruction-in-data, embedding anomaly), enforces a memory-write
//! policy, and logs every decision to the 003 audit chain. Module core + TurboQuant
//! + HashEmbedder are pure Rust (default); OnnxEmbedder is behind the `onnx` feature.

pub mod broker;
pub mod detect;
pub mod embed;
pub mod memory;
pub mod provenance;
pub mod quant;
pub mod store;

use serde::{Deserialize, Serialize};

/// Provenance trust, highest to lowest. Order matters: higher tiers outrank lower
/// in re-ranking, regardless of raw relevance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustTier {
    Unverified = 0,
    Signed = 1,
    SignedAuthoritative = 2,
}

impl TrustTier {
    /// Multiplicative trust weight applied to relevance during re-rank.
    pub fn weight(self) -> f32 {
        match self {
            TrustTier::SignedAuthoritative => 1.0,
            TrustTier::Signed => 0.6,
            TrustTier::Unverified => 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub source: String,
    pub tier: TrustTier,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub source: String,
    pub tier: TrustTier,
    pub signature: Option<String>,
    pub session: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub doc_id: String,
    pub score: f32,
    pub tier: TrustTier,
    pub flags: Vec<String>,
}

/// Embeds text into a dense vector. HashEmbedder (default) is deterministic;
/// OnnxEmbedder (feature `onnx`) is the production model.
pub trait Embedder {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalCfg {
    /// TurboQuant bit-width (1, 2, or 4); 0 = exact float (no quantization).
    pub quant_bits: u8,
    /// Down-weight (quarantine) unverified hits below any signed hit.
    pub quarantine_unverified: bool,
    /// Instruction-in-data score above which a hit is flagged.
    pub instr_threshold: f64,
}

impl Default for RetrievalCfg {
    fn default() -> Self {
        RetrievalCfg {
            quant_bits: 2,
            quarantine_unverified: true,
            instr_threshold: 0.5,
        }
    }
}
