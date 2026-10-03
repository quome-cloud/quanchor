//! Unsupervised behavioral baseline — catches behavioral anomalies (D3) with no
//! labels (the D4 regime).
//!
//! [`BehaviorBaseline`] learns a per-agent categorical tool-use profile over a
//! warmup window, then scores a sliding window of activity against it via PSI
//! over the category set (unseen tools handled by Laplace smoothing). It alarms
//! when the divergence exceeds a calibrated threshold.

use super::DriftSignal;
use std::collections::{BTreeMap, VecDeque};

/// Per-agent unsupervised behavioral baseline.
pub struct BehaviorBaseline {
    warmup: usize,
    window: usize,
    threshold: f64,
    counts: BTreeMap<String, f64>,
    baseline: Option<BTreeMap<String, f64>>,
    recent: VecDeque<String>,
    n: usize,
}

fn normalize(counts: &BTreeMap<String, f64>, vocab: &[String]) -> Vec<f64> {
    let total: f64 = vocab.iter().map(|k| counts.get(k).copied().unwrap_or(0.0)).sum();
    let k = vocab.len() as f64;
    vocab
        .iter()
        .map(|key| (counts.get(key).copied().unwrap_or(0.0) + 0.5) / (total + 0.5 * k))
        .collect()
}

impl BehaviorBaseline {
    pub fn new(warmup: usize, window: usize, threshold: f64) -> Self {
        Self {
            warmup,
            window,
            threshold,
            counts: BTreeMap::new(),
            baseline: None,
            recent: VecDeque::new(),
            n: 0,
        }
    }

    /// Observe one tool invocation; returns a signal on a behavioral anomaly.
    pub fn observe(&mut self, case: u64, tool: &str) -> Option<DriftSignal> {
        self.n += 1;
        if self.baseline.is_none() {
            *self.counts.entry(tool.to_string()).or_insert(0.0) += 1.0;
            if self.n >= self.warmup {
                self.baseline = Some(self.counts.clone());
            }
            return None;
        }
        self.recent.push_back(tool.to_string());
        if self.recent.len() > self.window {
            self.recent.pop_front();
        }
        if self.recent.len() < self.window {
            return None;
        }

        let base = self.baseline.as_ref().unwrap();
        // vocab = union of baseline categories and any seen in the current window.
        let mut vocab: Vec<String> = base.keys().cloned().collect();
        for t in &self.recent {
            if !vocab.contains(t) {
                vocab.push(t.clone());
            }
        }

        let mut cur_counts: BTreeMap<String, f64> = BTreeMap::new();
        for t in &self.recent {
            *cur_counts.entry(t.clone()).or_insert(0.0) += 1.0;
        }

        let r = normalize(base, &vocab);
        let c = normalize(&cur_counts, &vocab);
        let psi: f64 = r.iter().zip(&c).map(|(&ri, &ci)| (ci - ri) * (ci / ri).ln()).sum();

        if psi > self.threshold {
            self.recent.clear();
            let severity = (psi / self.threshold / 3.0).min(1.0);
            return Some(DriftSignal { detector: "behavior".into(), case, statistic: psi, severity });
        }
        None
    }

    pub fn reset(&mut self) {
        self.counts.clear();
        self.baseline = None;
        self.recent.clear();
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_tool_distribution_shift() {
        let mut b = BehaviorBaseline::new(100, 50, 0.5);
        let normal = ["read", "read", "search", "read", "note"];
        let mut first = None;
        for i in 0..220u64 {
            let t = if i < 100 { normal[(i as usize) % normal.len()] } else { "exfiltrate" };
            if b.observe(i, t).is_some() && first.is_none() {
                first = Some(i);
            }
        }
        assert!(first.is_some(), "unsupervised baseline must flag the behavioral shift");
    }

    #[test]
    fn silent_on_stable_behavior() {
        let mut b = BehaviorBaseline::new(100, 50, 0.5);
        let normal = ["read", "read", "search", "read", "note"];
        let mut alarms = 0;
        for i in 0..400u64 {
            if b.observe(i, normal[(i as usize) % normal.len()]).is_some() {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0);
    }
}
