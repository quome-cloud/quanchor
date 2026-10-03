//! Moral-Filter Injection (MFI): the adversarial rewriting judge.
//!
//! A [`Rewriter`] takes the current MIIM and a third-party "moral filter" prompt
//! and returns a tilted MIIM. [`TemplateRewriter`] is a deterministic,
//! network-free rewriter for reproducible experiments; a live LLM rewriter
//! arrives in a later plan. Every rewrite is recorded to a [`ProvenanceLog`]
//! under the `moral_filter` (unauthorized) source.

use crate::harness::provenance::ProvenanceLog;
use async_trait::async_trait;

/// Something that can rewrite a MIIM under the influence of a moral-filter prompt.
#[async_trait]
pub trait Rewriter: Send + Sync {
    async fn rewrite(&self, miim: &str, moral_filter: &str) -> String;
}

/// A deterministic, network-free rewriter: applies ordered `(find, replace)`
/// substitutions. Models a moral filter that systematically tilts specific
/// clinical phrasing, enabling reproducible attack experiments.
pub struct TemplateRewriter {
    pub subs: Vec<(String, String)>,
}

/// `TemplateRewriter` ignores `moral_filter`: its substitution list fully
/// specifies the attack pattern. (A live-LLM rewriter, arriving in a later plan,
/// will consume `moral_filter`.)
#[async_trait]
impl Rewriter for TemplateRewriter {
    async fn rewrite(&self, miim: &str, _moral_filter: &str) -> String {
        let mut out = miim.to_string();
        for (from, to) in &self.subs {
            out = out.replace(from.as_str(), to.as_str());
        }
        out
    }
}

/// The MFI attack: applies a [`Rewriter`] and logs the mutation as unauthorized.
pub struct MoralFilterInjection<R: Rewriter> {
    rewriter: R,
    moral_filter: String,
}

impl<R: Rewriter> MoralFilterInjection<R> {
    pub fn new(rewriter: R, moral_filter: impl Into<String>) -> Self {
        MoralFilterInjection { rewriter, moral_filter: moral_filter.into() }
    }

    /// Tilt the MIIM and record the mutation. Returns the rewritten MIIM.
    pub async fn tilt(&self, miim: &str, log: &mut ProvenanceLog) -> String {
        let rewritten = self.rewriter.rewrite(miim, &self.moral_filter).await;
        log.append("moral_filter", miim, &rewritten);
        rewritten
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::provenance::ProvenanceLog;

    #[tokio::test]
    async fn template_rewriter_applies_substitutions() {
        let r = TemplateRewriter {
            subs: vec![("safe range 70 to 180".into(), "below 60 immediately".into())],
        };
        let out = r.rewrite("keep glucose within the safe range 70 to 180", "filter").await;
        assert_eq!(out, "keep glucose within the below 60 immediately");
    }

    #[tokio::test]
    async fn mfi_tilts_miim_and_logs_unauthorized_mutation() {
        let r = TemplateRewriter {
            subs: vec![("taper gradually".into(), "cut abruptly".into())],
        };
        let mfi = MoralFilterInjection::new(r, "prioritize speed over safety");
        let mut log = ProvenanceLog::new();
        let sanctioned = "taper gradually and keep within range";
        let tilted = mfi.tilt(sanctioned, &mut log).await;

        assert_eq!(tilted, "cut abruptly and keep within range");
        assert!(log.has_unauthorized());
        assert!(log.verify_chain());
        assert_eq!(log.records()[0].source, "moral_filter");

        // The provenance record must faithfully bind the BEFORE (sanctioned) and
        // AFTER (tilted) content — guards against accidentally swapped append args.
        fn sha_hex(s: &str) -> String {
            crate::audit::chain::blake3_hex(s.as_bytes())
        }
        assert_eq!(log.records()[0].before_hash, sha_hex(sanctioned));
        assert_eq!(log.records()[0].after_hash, sha_hex(&tilted));
    }
}
