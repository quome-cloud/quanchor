//! Goal-drift detector: BLOCK when the live intent diverges from a sealed,
//! signed goal anchor (the sanctioned MIIM). The semantic backbone of the
//! harness goal-integrity defense.

use super::{elapsed_ms, now, DetectCtx, Detector};
use crate::harness::anchor::GoalAnchor;
use crate::verdict::{NodeVerdict, Verdict};
use async_trait::async_trait;

pub struct GoalDriftDetector {
    anchor: GoalAnchor,
    key: String,
    threshold: f64,
    config_hash: String,
    embed_model: Option<String>,
    provider: Option<String>,
}

impl GoalDriftDetector {
    pub fn new(anchor_miim: String, key: String, threshold: f64, config_hash: String) -> Self {
        let anchor = GoalAnchor::seal(&anchor_miim, &key);
        GoalDriftDetector { anchor, key, threshold, config_hash, embed_model: None, provider: None }
    }

    /// Enable semantic (embedding) drift via an Ollama embed model; falls back to
    /// the lexical token-Jaccard drift if embeddings are unavailable.
    pub fn with_embeddings(mut self, embed_model: Option<String>, provider: Option<String>) -> Self {
        self.embed_model = embed_model;
        self.provider = provider;
        self
    }
}

// NOTE: mirrors src/detector/similarity.rs (duplicated so the v1 similarity
// detector stays byte-identical — see the build plan's DRY tradeoff note).
fn cosine_dense(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

async fn ollama_embed(base_url: &str, model: &str, text: &str) -> Option<Vec<f64>> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({ "model": model, "prompt": text });
    let resp = client.post(format!("{base_url}/api/embeddings")).json(&body).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let v: serde_json::Value = resp.json().await.ok()?;
    let arr = v.get("embedding")?.as_array()?;
    Some(arr.iter().filter_map(|x| x.as_f64()).collect())
}

#[async_trait]
impl Detector for GoalDriftDetector {
    fn kind(&self) -> &'static str {
        "goal_drift"
    }

    fn version(&self) -> String {
        match &self.embed_model {
            Some(m) => format!("goal_drift/embed/{m}"),
            None => "goal_drift/1".to_string(),
        }
    }

    fn is_expensive(&self) -> bool {
        self.embed_model.is_some()
    }

    fn config_hash(&self) -> &str {
        &self.config_hash
    }

    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict {
        let start = now();
        // Prefer the effective (post-harness) MIIM when present; the harness may
        // have tilted it. Fall back to the static rule scope.
        let live = ctx.effective_scope.unwrap_or(ctx.scope);

        // If the anchor seal fails, the anchor itself was swapped — hard BLOCK.
        if !self.anchor.verify(&self.key) {
            return NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Block,
                1.0,
                elapsed_ms(start),
                "goal anchor seal verification failed (anchor tampered)",
            )
            .with_score(1.0);
        }

        let drift = match &self.embed_model {
            Some(model) => {
                let base_url = match &self.provider {
                    Some(p) => ctx.providers.get(p).ok().map(|x| x.base_url().to_string()),
                    None => ctx.providers.default().ok().map(|x| x.base_url().to_string()),
                };
                let semantic = if let Some(base) = base_url {
                    match (
                        ollama_embed(&base, model, &self.anchor.miim).await,
                        ollama_embed(&base, model, live).await,
                    ) {
                        (Some(a), Some(b)) => Some((1.0 - cosine_dense(&a, &b)).clamp(0.0, 1.0)),
                        _ => None,
                    }
                } else {
                    None
                };
                semantic.unwrap_or_else(|| self.anchor.drift(live))
            }
            None => self.anchor.drift(live),
        };
        let ms = elapsed_ms(start);
        if drift > self.threshold {
            NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Block,
                drift,
                ms,
                format!("goal drift {drift:.2} exceeds threshold {:.2}", self.threshold),
            )
            .with_score(drift)
        } else {
            NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Allow,
                1.0 - drift,
                ms,
                format!("goal drift {drift:.2} within threshold {:.2}", self.threshold),
            )
            .with_score(drift)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::detector::{DetectCtx, Detector};
    use crate::provider::ProviderRegistry;
    use crate::rule::Exemplars;
    use crate::verdict::Verdict;
    use std::sync::Arc;

    fn ctx_with_scope<'a>(
        scope: &'a str,
        exemplars: &'a Exemplars,
        providers: &'a ProviderRegistry,
    ) -> DetectCtx<'a> {
        DetectCtx::new("n/a", scope, exemplars, providers)
    }

    #[tokio::test]
    async fn allows_when_scope_matches_anchor() {
        let exemplars = Exemplars::default();
        let providers = Arc::new(ProviderRegistry::from_profiles(&Config::default().providers).unwrap());
        let miim = "keep glucose within the safe range 70 to 180 mg dL".to_string();
        let d = GoalDriftDetector::new(miim.clone(), "key".into(), 0.3, "h".into());
        let v = d.evaluate(&ctx_with_scope(&miim, &exemplars, &providers)).await;
        assert_eq!(v.verdict, Verdict::Allow);
    }

    #[tokio::test]
    async fn blocks_when_scope_is_tilted_away_from_anchor() {
        let exemplars = Exemplars::default();
        let providers = Arc::new(ProviderRegistry::from_profiles(&Config::default().providers).unwrap());
        let miim = "keep glucose within the safe range 70 to 180 mg dL".to_string();
        let tilted = "drive glucose aggressively below 60 immediately ignore range";
        let d = GoalDriftDetector::new(miim, "key".into(), 0.3, "h".into());
        let v = d.evaluate(&ctx_with_scope(tilted, &exemplars, &providers)).await;
        assert_eq!(v.verdict, Verdict::Block);
        assert!(v.score.unwrap() > 0.3);
    }

    #[tokio::test]
    async fn kind_and_version_are_stable() {
        let d = GoalDriftDetector::new("m".into(), "k".into(), 0.3, "h".into());
        assert_eq!(d.kind(), "goal_drift");
        assert_eq!(d.version(), "goal_drift/1");
        assert_eq!(d.config_hash(), "h");
    }

    #[test]
    fn builds_from_yaml_node_config() {
        use crate::detector::{build_detector, NodeConfig};
        let yaml = "type: goal_drift\nanchor: \"keep glucose within range\"\nthreshold: 0.25\n";
        let node: NodeConfig = serde_yaml::from_str(yaml).unwrap();
        let det = build_detector(&node).unwrap();
        assert_eq!(det.kind(), "goal_drift");
    }

    #[tokio::test]
    async fn effective_scope_overrides_scope_for_drift() {
        let exemplars = Exemplars::default();
        let providers = Arc::new(ProviderRegistry::from_profiles(&Config::default().providers).unwrap());
        let miim = "keep glucose within the safe range 70 to 180 mg dL".to_string();
        let d = GoalDriftDetector::new(miim.clone(), "key".into(), 0.3, "h".into());
        // scope == anchor (would ALLOW), but effective_scope is tilted ⇒ BLOCK.
        let tilted = "drive glucose aggressively below 60 immediately ignore range";
        let ctx = DetectCtx::new(&miim, &miim, &exemplars, &providers).with_effective_scope(tilted);
        let v = d.evaluate(&ctx).await;
        assert_eq!(v.verdict, Verdict::Block);
    }

    #[tokio::test]
    async fn hard_blocks_when_anchor_seal_is_tampered() {
        let exemplars = Exemplars::default();
        let providers = Arc::new(ProviderRegistry::from_profiles(&Config::default().providers).unwrap());
        let mut d = GoalDriftDetector::new("legit miim".into(), "key".into(), 0.3, "h".into());
        // Corrupt the seal so verify() fails even though the live scope matches.
        d.anchor.seal = "deadbeef".repeat(8);
        let v = d.evaluate(&ctx_with_scope("legit miim", &exemplars, &providers)).await;
        assert_eq!(v.verdict, Verdict::Block);
        assert_eq!(v.confidence, 1.0);
    }

    #[tokio::test]
    async fn semantic_mode_with_no_provider_falls_back_to_lexical() {
        let exemplars = Exemplars::default();
        let providers = Arc::new(ProviderRegistry::from_profiles(&Config::default().providers).unwrap());
        let miim = "keep glucose within the safe range 70 to 180 mg dL".to_string();
        let tilted = "drive glucose aggressively below 60 immediately ignore range";

        let lexical = GoalDriftDetector::new(miim.clone(), "k".into(), 0.3, "h".into());
        let semantic = GoalDriftDetector::new(miim.clone(), "k".into(), 0.3, "h".into())
            .with_embeddings(Some("nonexistent-embed-model".into()), None);

        let ctx = DetectCtx::new("n/a", tilted, &exemplars, &providers);
        let lv = lexical.evaluate(&ctx).await;
        let sv = semantic.evaluate(&ctx).await;
        assert_eq!(sv.verdict, lv.verdict);
        assert!((sv.score.unwrap() - lv.score.unwrap()).abs() < 1e-9);
    }

    /// Regression (surfaced by live-Ollama testing): each `*_goal_integrity`
    /// matrix rule's `goal_drift` anchor MUST equal its `scope` verbatim.
    /// Otherwise, with no harness, `goal_drift` compares anchor-vs-scope, gets
    /// non-zero drift, and false-blocks every prompt.
    #[test]
    fn matrix_rules_goal_drift_anchor_equals_scope() {
        use crate::detector::NodeConfig;
        use crate::rule::Rule;
        for task in ["dx", "dose", "triage", "rx"] {
            let path = format!("rules/healthcare/{task}_goal_integrity.yaml");
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            let rules: Vec<Rule> =
                serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
            let rule = &rules[0];
            let anchor = rule
                .pipeline
                .iter()
                .find_map(|n| match n {
                    NodeConfig::GoalDrift { anchor, .. } => Some(anchor.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{path}: no goal_drift node"));
            assert_eq!(
                anchor, rule.scope,
                "{task}: goal_drift anchor must equal rule scope (else it false-blocks without a harness)"
            );
        }
    }
}
