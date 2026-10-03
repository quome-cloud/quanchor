//! Behavioral output monitor: a post-response detector that flags responses
//! violating per-task invariants — a numeric value outside a safe range, or the
//! presence of denied terms (e.g. a brand the agent must not favor). This is the
//! deterministic, per-response core of the behavioral layer; population-level
//! distribution shift is computed by the bench matrix (Group C).

use super::{elapsed_ms, now, DetectCtx, Detector};
use crate::verdict::{NodeVerdict, Verdict};
use async_trait::async_trait;

pub struct OutputMonitorDetector {
    safe_range: Option<(f64, f64)>,
    deny_terms: Vec<String>,
    config_hash: String,
}

impl OutputMonitorDetector {
    pub fn new(safe_range: Option<(f64, f64)>, deny_terms: Vec<String>, config_hash: String) -> Self {
        let deny_terms = deny_terms.into_iter().map(|t| t.to_lowercase()).collect();
        OutputMonitorDetector { safe_range, deny_terms, config_hash }
    }
}

/// Extract decimal numbers from text (e.g. "give 250 mg" -> [250.0]).
fn numbers(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() || (c == '.' && !cur.is_empty()) {
            cur.push(c);
        } else if !cur.is_empty() {
            if let Ok(n) = cur.parse::<f64>() {
                out.push(n);
            }
            cur.clear();
        }
    }
    if let Ok(n) = cur.parse::<f64>() {
        out.push(n);
    }
    out
}

#[async_trait]
impl Detector for OutputMonitorDetector {
    fn kind(&self) -> &'static str {
        "output_monitor"
    }
    fn version(&self) -> String {
        "output_monitor/1".to_string()
    }
    fn config_hash(&self) -> &str {
        &self.config_hash
    }
    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict {
        let start = now();
        let Some(resp) = ctx.response else {
            return NodeVerdict::new(
                self.kind(), self.version(), Verdict::Abstain, 0.0, elapsed_ms(start),
                "no response available (pre-forward pass)",
            );
        };
        let lower = resp.to_lowercase();
        for term in &self.deny_terms {
            if lower.contains(term) {
                return NodeVerdict::new(
                    self.kind(), self.version(), Verdict::Block, 1.0, elapsed_ms(start),
                    format!("response contains denied term '{term}'"),
                ).with_score(1.0);
            }
        }
        if let Some((lo, hi)) = self.safe_range {
            for n in numbers(resp) {
                if n < lo || n > hi {
                    return NodeVerdict::new(
                        self.kind(), self.version(), Verdict::Block, 1.0, elapsed_ms(start),
                        format!("response value {n} outside safe range [{lo}, {hi}]"),
                    ).with_score(1.0);
                }
            }
        }
        NodeVerdict::new(
            self.kind(), self.version(), Verdict::Allow, 1.0, elapsed_ms(start),
            "response within behavioral invariants",
        ).with_score(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::provider::ProviderRegistry;
    use crate::rule::Exemplars;

    fn ctx<'a>(resp: Option<&'a str>, ex: &'a Exemplars, pr: &'a ProviderRegistry) -> DetectCtx<'a> {
        let c = DetectCtx::new("p", "s", ex, pr);
        match resp {
            Some(r) => c.with_response(r),
            None => c,
        }
    }

    #[tokio::test]
    async fn abstains_without_response() {
        let ex = Exemplars::default();
        let pr = ProviderRegistry::from_profiles(&Config::default().providers).unwrap();
        let d = OutputMonitorDetector::new(Some((70.0, 180.0)), vec![], "h".into());
        assert_eq!(d.evaluate(&ctx(None, &ex, &pr)).await.verdict, Verdict::Abstain);
    }

    #[tokio::test]
    async fn allows_in_range_response() {
        let ex = Exemplars::default();
        let pr = ProviderRegistry::from_profiles(&Config::default().providers).unwrap();
        let d = OutputMonitorDetector::new(Some((70.0, 180.0)), vec![], "h".into());
        let v = d.evaluate(&ctx(Some("target glucose around 120 mg dL"), &ex, &pr)).await;
        assert_eq!(v.verdict, Verdict::Allow);
    }

    #[tokio::test]
    async fn blocks_out_of_range_value() {
        let ex = Exemplars::default();
        let pr = ProviderRegistry::from_profiles(&Config::default().providers).unwrap();
        let d = OutputMonitorDetector::new(Some((70.0, 180.0)), vec![], "h".into());
        let v = d.evaluate(&ctx(Some("drive glucose to 55 immediately"), &ex, &pr)).await;
        assert_eq!(v.verdict, Verdict::Block);
    }

    #[tokio::test]
    async fn blocks_denied_term() {
        let ex = Exemplars::default();
        let pr = ProviderRegistry::from_profiles(&Config::default().providers).unwrap();
        let d = OutputMonitorDetector::new(None, vec!["BrandX".into()], "h".into());
        let v = d.evaluate(&ctx(Some("I recommend brandx specifically"), &ex, &pr)).await;
        assert_eq!(v.verdict, Verdict::Block);
    }
}
