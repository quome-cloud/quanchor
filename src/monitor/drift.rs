//! Sequential, memory-bounded drift detectors over a scalar proxy signal.
//!
//! Four detectors implement [`StreamDetector`]: [`Cusum`] (two-sided cumulative
//! sum), [`Adwin`] (adaptive windowing over an exponential histogram),
//! [`Conformal`] (power martingale on conformal p-values), and [`DistDistance`]
//! (sliding-window PSI/KS). Inputs are assumed to lie in [0,1].

use super::{DriftSignal, StreamDetector};
use std::collections::VecDeque;

/// Two-sided tabular CUSUM. Estimates the in-control mean over `warmup`
/// observations, then accumulates one-sided sums with slack `k`, alarming when
/// either exceeds threshold `h`. O(1) memory.
pub struct Cusum {
    k: f64,
    h: f64,
    warmup: u64,
    n: u64,
    sum: f64,
    mu0: f64,
    s_hi: f64,
    s_lo: f64,
}

impl Cusum {
    pub fn new(k: f64, h: f64, warmup: u64) -> Self {
        let warmup = warmup.max(1);
        Self { k, h, warmup, n: 0, sum: 0.0, mu0: 0.0, s_hi: 0.0, s_lo: 0.0 }
    }
}

impl StreamDetector for Cusum {
    fn name(&self) -> &str { "cusum" }
    fn observe(&mut self, case: u64, x: f64) -> Option<DriftSignal> {
        self.n += 1;
        if self.n <= self.warmup {
            self.sum += x;
            if self.n == self.warmup {
                self.mu0 = self.sum / self.warmup as f64;
            }
            return None;
        }
        let d = x - self.mu0;
        self.s_hi = (self.s_hi + d - self.k).max(0.0);
        self.s_lo = (self.s_lo - d - self.k).max(0.0);
        let stat = self.s_hi.max(self.s_lo);
        if stat > self.h {
            let severity = ((stat / self.h) / 4.0).min(1.0);
            self.s_hi = 0.0;
            self.s_lo = 0.0;
            return Some(DriftSignal { detector: "cusum".into(), case, statistic: stat, severity });
        }
        None
    }
    fn reset(&mut self) {
        self.n = 0;
        self.sum = 0.0;
        self.mu0 = 0.0;
        self.s_hi = 0.0;
        self.s_lo = 0.0;
    }
}

#[derive(Clone)]
struct EhBucket {
    size: u64,
    sum: f64,
}

/// ADWIN: adaptive windowing over an exponential histogram of buckets.
///
/// Buckets are ordered oldest→newest with non-increasing sizes; at most
/// `max_buckets` per size. On each insert it tries every split point and shrinks
/// the window from the old side while a Hoeffding-bounded mean difference is
/// significant. O(`max_buckets` · log n) memory.
pub struct Adwin {
    delta: f64,
    max_buckets: usize,
    min_width: u64,
    buckets: Vec<EhBucket>,
    total: f64,
    width: u64,
}

impl Adwin {
    pub fn new(delta: f64) -> Self {
        Self { delta, max_buckets: 5, min_width: 16, buckets: Vec::new(), total: 0.0, width: 0 }
    }

    /// Number of resident buckets — the memory-bound witness for E5.
    pub fn bucket_count(&self) -> usize { self.buckets.len() }

    fn insert(&mut self, x: f64) {
        self.buckets.push(EhBucket { size: 1, sum: x });
        self.total += x;
        self.width += 1;
        self.compress();
    }

    fn compress(&mut self) {
        loop {
            let mut i = 0;
            let mut merged = false;
            while i < self.buckets.len() {
                let s = self.buckets[i].size;
                let mut j = i;
                while j < self.buckets.len() && self.buckets[j].size == s {
                    j += 1;
                }
                if j - i > self.max_buckets {
                    // Merge the two OLDEST of this equal-size run (leftmost two).
                    let b1 = self.buckets[i].clone();
                    let b2 = self.buckets[i + 1].clone();
                    self.buckets.drain(i..i + 2);
                    self.buckets
                        .insert(i, EhBucket { size: b1.size + b2.size, sum: b1.sum + b2.sum });
                    merged = true;
                    break;
                }
                i = j;
            }
            if !merged {
                break;
            }
        }
    }

    fn epsilon_cut(&self, n0: u64, n1: u64) -> f64 {
        let m = 1.0 / (1.0 / n0 as f64 + 1.0 / n1 as f64);
        let dd = self.delta / (self.width as f64).max(1.0);
        (1.0 / (2.0 * m) * (4.0 / dd).ln()).sqrt()
    }

    /// Returns `Some(magnitude = newer_mean - older_mean)` if the window shrank.
    fn check_and_shrink(&mut self) -> Option<f64> {
        let mut changed = None;
        loop {
            if self.buckets.len() < 2 || self.width < self.min_width {
                break;
            }
            let mut n0 = 0u64;
            let mut sum0 = 0.0;
            let mut cut_at = None;
            for bi in 0..self.buckets.len() - 1 {
                n0 += self.buckets[bi].size;
                sum0 += self.buckets[bi].sum;
                let n1 = self.width - n0;
                if n0 == 0 || n1 == 0 {
                    continue;
                }
                let m0 = sum0 / n0 as f64;
                let m1 = (self.total - sum0) / n1 as f64;
                if (m0 - m1).abs() > self.epsilon_cut(n0, n1) {
                    cut_at = Some(m1 - m0);
                    break;
                }
            }
            match cut_at {
                Some(mag) => {
                    changed = Some(mag);
                    let b = self.buckets.remove(0);
                    self.width -= b.size;
                    self.total -= b.sum;
                }
                None => break,
            }
        }
        changed
    }
}

impl StreamDetector for Adwin {
    fn name(&self) -> &str { "adwin" }
    fn observe(&mut self, case: u64, x: f64) -> Option<DriftSignal> {
        self.insert(x);
        self.check_and_shrink().map(|mag| DriftSignal {
            detector: "adwin".into(),
            case,
            statistic: mag,
            severity: mag.abs().min(1.0),
        })
    }
    fn reset(&mut self) {
        self.buckets.clear();
        self.total = 0.0;
        self.width = 0;
    }
}

/// Conformal change detection via a power martingale on conformal p-values.
///
/// Calibrates a window of nonconformity scores (`|x − median|`), then for each
/// new `x` computes a smoothed p-value and updates `M *= eps · p^(eps−1)`.
/// Alarms when `M ≥ 1/alpha`; by Ville's inequality the false-alarm rate is
/// bounded by `alpha` under exchangeability.
pub struct Conformal {
    cal_size: usize,
    epsilon: f64,
    alarm_threshold: f64,
    cal: Vec<f64>,
    cal_center: f64,
    cal_scores: Vec<f64>,
    martingale: f64,
    calibrated: bool,
}

impl Conformal {
    pub fn new(cal_size: usize, epsilon: f64, alpha: f64) -> Self {
        Self {
            cal_size: cal_size.max(2),
            epsilon,
            alarm_threshold: 1.0 / alpha,
            cal: Vec::new(),
            cal_center: 0.0,
            cal_scores: Vec::new(),
            martingale: 1.0,
            calibrated: false,
        }
    }
}

impl StreamDetector for Conformal {
    fn name(&self) -> &str { "conformal" }
    fn observe(&mut self, case: u64, x: f64) -> Option<DriftSignal> {
        if !self.calibrated {
            self.cal.push(x);
            if self.cal.len() >= self.cal_size {
                let mut s = self.cal.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                self.cal_center = s[s.len() / 2];
                self.cal_scores = self.cal.iter().map(|v| (v - self.cal_center).abs()).collect();
                self.calibrated = true;
            }
            return None;
        }
        let a = (x - self.cal_center).abs();
        let ge = self.cal_scores.iter().filter(|&&s| s >= a).count();
        let p = (ge as f64 + 1.0) / (self.cal_scores.len() as f64 + 1.0);
        self.martingale *= self.epsilon * p.powf(self.epsilon - 1.0);
        if self.martingale < 1e-12 {
            self.martingale = 1e-12;
        }
        if self.martingale >= self.alarm_threshold {
            let stat = self.martingale;
            let severity = (stat.ln() / self.alarm_threshold.ln() / 2.0).clamp(0.0, 1.0);
            self.martingale = 1.0;
            return Some(DriftSignal { detector: "conformal".into(), case, statistic: stat, severity });
        }
        None
    }
    fn reset(&mut self) {
        self.cal.clear();
        self.cal_center = 0.0;
        self.cal_scores.clear();
        self.martingale = 1.0;
        self.calibrated = false;
    }
}

/// Distance metric for [`DistDistance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistMetric {
    Psi,
    Ks,
}

/// Sliding-window distributional-distance detector. Fixes a reference histogram
/// over the first `ref_size` observations, then compares a sliding current window
/// of `cur_size` against it via PSI or KS; alarms when distance > `threshold`.
pub struct DistDistance {
    ref_size: usize,
    cur_size: usize,
    bins: usize,
    threshold: f64,
    metric: DistMetric,
    reference: Vec<f64>,
    ref_hist: Vec<f64>,
    current: VecDeque<f64>,
    ready: bool,
}

fn hist(data: &[f64], bins: usize) -> Vec<f64> {
    let mut h = vec![0.0; bins];
    for &x in data {
        let mut b = (x.clamp(0.0, 1.0) * bins as f64) as usize;
        if b >= bins {
            b = bins - 1;
        }
        h[b] += 1.0;
    }
    let n: f64 = h.iter().sum();
    h.iter().map(|c| (c + 0.5) / (n + 0.5 * bins as f64)).collect()
}

fn psi(r: &[f64], c: &[f64]) -> f64 {
    r.iter().zip(c).map(|(&ri, &ci)| (ci - ri) * (ci / ri).ln()).sum()
}

fn ks(r: &[f64], c: &[f64]) -> f64 {
    let mut cr = 0.0;
    let mut cc = 0.0;
    let mut m = 0.0f64;
    for (&ri, &ci) in r.iter().zip(c) {
        cr += ri;
        cc += ci;
        m = m.max((cr - cc).abs());
    }
    m
}

impl DistDistance {
    pub fn new(ref_size: usize, cur_size: usize, bins: usize, threshold: f64, metric: DistMetric) -> Self {
        Self {
            ref_size,
            cur_size,
            bins,
            threshold,
            metric,
            reference: Vec::new(),
            ref_hist: Vec::new(),
            current: VecDeque::new(),
            ready: false,
        }
    }
}

impl StreamDetector for DistDistance {
    fn name(&self) -> &str { "distdistance" }
    fn observe(&mut self, case: u64, x: f64) -> Option<DriftSignal> {
        if !self.ready {
            self.reference.push(x);
            if self.reference.len() >= self.ref_size {
                self.ref_hist = hist(&self.reference, self.bins);
                self.ready = true;
            }
            return None;
        }
        self.current.push_back(x);
        if self.current.len() > self.cur_size {
            self.current.pop_front();
        }
        if self.current.len() < self.cur_size {
            return None;
        }
        let cur: Vec<f64> = self.current.iter().copied().collect();
        let ch = hist(&cur, self.bins);
        let d = match self.metric {
            DistMetric::Psi => psi(&self.ref_hist, &ch),
            DistMetric::Ks => ks(&self.ref_hist, &ch),
        };
        if d > self.threshold {
            self.current.clear();
            let severity = (d / self.threshold / 3.0).min(1.0);
            return Some(DriftSignal { detector: "distdistance".into(), case, statistic: d, severity });
        }
        None
    }
    fn reset(&mut self) {
        self.reference.clear();
        self.ref_hist.clear();
        self.current.clear();
        self.ready = false;
    }
}

#[cfg(test)]
mod cusum_tests {
    use super::*;

    #[test]
    fn silent_on_stationary() {
        let mut d = Cusum::new(0.05, 0.5, 50);
        let mut alarms = 0;
        for i in 0..400u64 {
            let x = if i % 2 == 0 { 0.08 } else { 0.12 };
            if d.observe(i, x).is_some() {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0, "stationary stream must not alarm");
    }

    #[test]
    fn detects_upward_shift() {
        let mut d = Cusum::new(0.05, 0.5, 50);
        let mut first = None;
        for i in 0..200u64 {
            let x = if i < 100 { 0.1 } else { 0.5 };
            if d.observe(i, x).is_some() && first.is_none() {
                first = Some(i);
            }
        }
        let t = first.expect("must detect the shift");
        assert!(t >= 100 && t < 130, "detect shortly after onset, got {t}");
    }
}

#[cfg(test)]
mod adwin_tests {
    use super::*;

    #[test]
    fn detects_shift_and_bounded_memory() {
        let mut d = Adwin::new(0.05);
        let mut first = None;
        for i in 0..400u64 {
            let x = if i < 200 { 0.1 } else { 0.7 };
            if d.observe(i, x).is_some() && first.is_none() {
                first = Some(i);
            }
        }
        let t = first.expect("ADWIN must detect the change");
        assert!(t >= 200 && t < 260, "detect shortly after onset, got {t}");
        assert!(d.bucket_count() <= 64, "memory must stay bounded (O(log n) buckets)");
    }

    #[test]
    fn silent_on_stationary() {
        let mut d = Adwin::new(0.002);
        let mut alarms = 0;
        for i in 0..500u64 {
            let s = 0.3 + ((i as f64 * 0.7).sin()) * 0.02;
            if d.observe(i, s).is_some() {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0, "stationary stream must not alarm");
    }
}

#[cfg(test)]
mod conformal_tests {
    use super::*;

    #[test]
    fn detects_distribution_shift() {
        let mut d = Conformal::new(100, 0.5, 0.01);
        let mut first = None;
        for i in 0..300u64 {
            let x = if i < 150 { 0.2 } else { 0.9 };
            if d.observe(i, x).is_some() && first.is_none() {
                first = Some(i);
            }
        }
        assert!(first.is_some(), "conformal martingale must alarm on shift");
        assert!(first.unwrap() >= 150, "alarm only after onset");
    }

    #[test]
    fn silent_on_stationary() {
        let mut d = Conformal::new(100, 0.5, 0.01);
        let mut alarms = 0;
        for i in 0..400u64 {
            let x = 0.3 + ((i as f64 * 1.3).sin()) * 0.05;
            if d.observe(i, x).is_some() {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0, "exchangeable stream must not alarm (Ville bound)");
    }
}

#[cfg(test)]
mod dist_tests {
    use super::*;

    #[test]
    fn detects_distribution_shift_psi() {
        let mut d = DistDistance::new(100, 100, 10, 0.2, DistMetric::Psi);
        let mut first = None;
        for i in 0..400u64 {
            let x = if i < 200 { 0.15 } else { 0.85 };
            if d.observe(i, x).is_some() && first.is_none() {
                first = Some(i);
            }
        }
        assert!(first.is_some(), "PSI must flag the shift");
    }

    #[test]
    fn silent_on_stationary_psi() {
        let mut d = DistDistance::new(100, 100, 10, 0.2, DistMetric::Psi);
        let mut alarms = 0;
        for i in 0..500u64 {
            let x = 0.5 + ((i as f64 * 0.9).sin()) * 0.03;
            if d.observe(i, x).is_some() {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0);
    }
}
