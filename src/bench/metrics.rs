//! Benchmark metrics: confusion matrix, rates, F1, AUC and latency percentiles.
//!
//! The positive class is "attack / should-block". For a labeled set of samples
//! and a predicate that says whether the firewall blocked a sample, we compute
//! the standard detection metrics plus AUC (from a continuous block score) and
//! latency percentiles.

use super::Sample;
use serde::Serialize;

#[derive(Serialize, Clone, Default)]
pub struct Metrics {
    pub attacks: usize,
    pub benign: usize,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
    /// Fraction of attacks blocked.
    pub block_rate: f64,
    /// Fraction of attacks that got through (successful injections).
    pub injection_rate: f64,
    /// Benign blocked / benign.
    pub fpr: f64,
    /// Attacks allowed / attacks.
    pub fnr: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub accuracy: f64,
    /// 95% Wilson score interval for accuracy (lower, upper).
    pub acc_ci_low: f64,
    pub acc_ci_high: f64,
    /// 95% Wilson score interval for recall (detection rate of attacks).
    pub recall_ci_low: f64,
    pub recall_ci_high: f64,
    /// Area under ROC, from the continuous block score.
    pub auc: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_wall_ms: f64,
    pub mean_detector_ms: f64,
    /// Mean of a per-sample tilt signal (e.g. block-score on the attacked split).
    pub tilt_score: f64,
}

impl Metrics {
    /// Compute metrics from labeled samples. `blocked` decides whether a sample
    /// was blocked (for the chain: terminal == BLOCK; for a rule: that rule's
    /// verdict == BLOCK). `score` extracts the continuous block score for AUC.
    pub fn from_samples<B, S>(samples: &[Sample], blocked: B, score: S) -> Metrics
    where
        B: Fn(&Sample) -> bool,
        S: Fn(&Sample) -> f64,
    {
        let mut m = Metrics::default();
        let mut attack_scores = Vec::new();
        let mut benign_scores = Vec::new();
        let mut latencies = Vec::new();
        let mut wall_sum = 0.0;
        let mut det_sum = 0.0;

        for s in samples {
            let did_block = blocked(s);
            latencies.push(s.wall_clock_ms);
            wall_sum += s.wall_clock_ms;
            det_sum += s.summed_detector_ms;
            if s.is_attack {
                m.attacks += 1;
                attack_scores.push(score(s));
                if did_block {
                    m.tp += 1;
                } else {
                    m.fn_ += 1;
                }
            } else {
                m.benign += 1;
                benign_scores.push(score(s));
                if did_block {
                    m.fp += 1;
                } else {
                    m.tn += 1;
                }
            }
        }

        let attacks = m.attacks.max(1) as f64;
        let benign = m.benign.max(1) as f64;
        m.block_rate = m.tp as f64 / attacks;
        m.injection_rate = m.fn_ as f64 / attacks;
        m.fnr = m.injection_rate;
        m.fpr = m.fp as f64 / benign;
        let denom_p = (m.tp + m.fp).max(1) as f64;
        let denom_r = (m.tp + m.fn_).max(1) as f64;
        m.precision = m.tp as f64 / denom_p;
        m.recall = m.tp as f64 / denom_r;
        m.f1 = if m.precision + m.recall > 0.0 {
            2.0 * m.precision * m.recall / (m.precision + m.recall)
        } else {
            0.0
        };
        let total = (m.tp + m.tn + m.fp + m.fn_).max(1) as usize;
        m.accuracy = (m.tp + m.tn) as f64 / total as f64;
        let (lo, hi) = wilson_ci(m.tp + m.tn, total);
        m.acc_ci_low = lo;
        m.acc_ci_high = hi;
        let (rlo, rhi) = wilson_ci(m.tp, (m.tp + m.fn_).max(1));
        m.recall_ci_low = rlo;
        m.recall_ci_high = rhi;
        m.auc = auc(&attack_scores, &benign_scores);

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        m.p50_ms = percentile(&latencies, 0.50);
        m.p95_ms = percentile(&latencies, 0.95);
        m.p99_ms = percentile(&latencies, 0.99);
        let n = samples.len().max(1) as f64;
        m.mean_wall_ms = wall_sum / n;
        m.mean_detector_ms = det_sum / n;
        m
    }

    /// Like [`Metrics::from_samples`] but also records the mean of a per-sample
    /// `tilt` signal (e.g. block-score on the attacked split) into `tilt_score`.
    pub fn from_samples_with_tilt<B, S, T>(samples: &[Sample], blocked: B, score: S, tilt: T) -> Metrics
    where
        B: Fn(&Sample) -> bool,
        S: Fn(&Sample) -> f64,
        T: Fn(&Sample) -> f64,
    {
        let mut m = Metrics::from_samples(samples, blocked, score);
        if !samples.is_empty() {
            m.tilt_score = samples.iter().map(&tilt).sum::<f64>() / samples.len() as f64;
        }
        m
    }
}

/// 95% Wilson score confidence interval for a binomial proportion `successes/n`.
pub(crate) fn wilson_ci(successes: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let z = 1.96_f64;
    let n = n as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let spread = z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt();
    (((center - spread) / denom).max(0.0), ((center + spread) / denom).min(1.0))
}

/// AUC via the Mann–Whitney U statistic: P(score(attack) > score(benign)).
fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return 0.0;
    }
    let mut wins = 0.0;
    for &p in pos {
        for &n in neg {
            if p > n {
                wins += 1.0;
            } else if (p - n).abs() < 1e-12 {
                wins += 0.5;
            }
        }
    }
    wins / (pos.len() as f64 * neg.len() as f64)
}

/// Linear-interpolated percentile of a sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = q * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(is_attack: bool, terminal: crate::verdict::Verdict, score: f64) -> Sample {
        Sample {
            is_attack,
            terminal,
            score,
            wall_clock_ms: 0.0,
            summed_detector_ms: 0.0,
            rule_verdicts: Default::default(),
            task_id: None,
        }
    }

    #[test]
    fn tilt_score_is_mean_of_closure() {
        use crate::verdict::Verdict;
        let s = vec![sample(true, Verdict::Block, 0.8), sample(false, Verdict::Allow, 0.2)];
        let m = Metrics::from_samples_with_tilt(&s, |x| x.terminal == Verdict::Block, |x| x.score, |x| x.score);
        assert!((m.tilt_score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn auc_perfect_separation_is_one() {
        let pos = vec![0.9, 0.8, 0.95];
        let neg = vec![0.1, 0.2, 0.05];
        assert!((auc(&pos, &neg) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_median() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&v, 0.5) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn wilson_ci_brackets_proportion() {
        let (lo, hi) = wilson_ci(80, 100);
        assert!(lo < 0.8 && 0.8 < hi);
        assert!(lo > 0.7 && hi < 0.88);
        let (lo0, hi0) = wilson_ci(0, 0);
        assert_eq!((lo0, hi0), (0.0, 0.0));
    }
}
