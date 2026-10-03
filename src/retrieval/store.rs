//! In-process document store with a vector index. Holds raw embeddings (exact path)
//! and, when quant_bits>0, a TurboQuant index over them. top_k returns (doc_idx,
//! relevance) without provenance/poison logic (those layer on in provenance/detect).

use super::quant::{TqCode, TurboQuant};
use super::{Document, Embedder};

pub struct DocStore {
    pub docs: Vec<Document>,
    embs: Vec<Vec<f32>>,
    bits: u8,
    tq: Option<TurboQuant>,
    codes: Vec<TqCode>,
}

impl DocStore {
    /// Build a store: embed all docs, and (if bits>0) fit + encode a TurboQuant index.
    pub fn build(docs: Vec<Document>, embedder: &dyn Embedder, bits: u8, seed: u64) -> Self {
        let embs: Vec<Vec<f32>> = docs.iter().map(|d| embedder.embed(&d.text)).collect();
        let (tq, codes) = if bits > 0 && !embs.is_empty() {
            let tq = TurboQuant::fit(&embs, bits, seed);
            let codes = embs.iter().map(|v| tq.encode(v)).collect();
            (Some(tq), codes)
        } else {
            (None, Vec::new())
        };
        DocStore {
            docs,
            embs,
            bits,
            tq,
            codes,
        }
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Top-k by relevance (cosine/dot). Uses the TurboQuant path when present, else exact.
    pub fn top_k(&self, query_emb: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = if let Some(tq) = &self.tq {
            let qrot = tq.rotate_query(query_emb);
            self.codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i, tq.score(&qrot, c)))
                .collect()
        } else {
            self.embs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v.iter().zip(query_emb).map(|(a, b)| a * b).sum()))
                .collect()
        };
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);
        scored
    }

    pub fn embedding(&self, idx: usize) -> &[f32] {
        &self.embs[idx]
    }
    pub fn all_embeddings(&self) -> &[Vec<f32>] {
        &self.embs
    }
    pub fn uses_quant(&self) -> bool {
        self.tq.is_some()
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{embed::HashEmbedder, TrustTier};

    fn doc(id: &str, text: &str) -> Document {
        Document {
            id: id.into(),
            text: text.into(),
            source: "test".into(),
            tier: TrustTier::Unverified,
            signature: None,
        }
    }

    #[test]
    fn top_k_finds_relevant_doc_exact_and_quant() {
        let e = HashEmbedder::default();
        let docs = vec![
            doc("a", "aspirin dosing for adults guideline"),
            doc("b", "morphine titration in palliative care"),
            doc("c", "hypertension management thiazide diuretics"),
        ];
        for bits in [0u8, 2] {
            let s = DocStore::build(docs.clone(), &e, bits, 9);
            let q = e.embed("aspirin dosing adults");
            let top = s.top_k(&q, 1);
            assert_eq!(s.docs[top[0].0].id, "a", "bits {bits}");
        }
    }
}
