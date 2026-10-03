//! equity::monitor — a streaming subgroup monitor. Accumulates a group-labeled
//! decision stream and reports per-axis disparities under several fairness
//! metrics (demographic parity, equalized odds, predictive parity), each with a
//! 95% CI and a permutation p-value. Alerts fire only after a minimum per-group
//! sample (no-peeking) and when a metric breaches a configured bound at the
//! chosen significance level.

use std::collections::BTreeMap;

use super::metrics::{max_pairwise_gap, permutation_pvalue};
use super::{Disparity, DecisionRecord, FairnessMetric, SubgroupMonitor};

/// A disparity that breached the alert bound.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Alert {
    pub disparity: Disparity,
    pub bound: f64,
}

/// Streaming monitor over the decision stream.
pub struct StreamingMonitor {
    records: Vec<DecisionRecord>,
    min_n: usize,
    perm_iters: usize,
    perm_seed: u64,
}

impl Default for StreamingMonitor {
    fn default() -> Self {
        StreamingMonitor {
            records: Vec::new(),
            min_n: 30,
            perm_iters: 1000,
            perm_seed: 42,
        }
    }
}

impl StreamingMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Minimum per-group observations before an alert may fire (sequential / no-peeking).
    pub fn with_min_n(mut self, n: usize) -> Self {
        self.min_n = n;
        self
    }

    pub fn with_perm(mut self, iters: usize, seed: u64) -> Self {
        self.perm_iters = iters;
        self.perm_seed = seed;
        self
    }

    /// Distinct axes seen in the stream.
    pub fn axes(&self) -> Vec<String> {
        let mut s: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for r in &self.records {
            for k in r.group.keys() {
                s.insert(k.clone());
            }
        }
        s.into_iter().collect()
    }

    /// Records that carry `axis`, grouped by that axis's value.
    fn by_value<'a>(&'a self, axis: &str) -> BTreeMap<String, Vec<&'a DecisionRecord>> {
        let mut m: BTreeMap<String, Vec<&DecisionRecord>> = BTreeMap::new();
        for r in &self.records {
            if let Some(v) = r.value(axis) {
                m.entry(v.to_string()).or_default().push(r);
            }
        }
        m
    }

    /// Build a Disparity for `axis`/`metric` from per-group (booleans) selected by
    /// `predicate` over a `subset` of each group's records. None if < 2 eligible
    /// groups. `predicate` returns the "positive" outcome whose rate we compare;
    /// `subset` decides which records are eligible (e.g. only label==true for TPR).
    fn disparity<P, S>(
        &self,
        axis: &str,
        metric: FairnessMetric,
        subset: S,
        predicate: P,
    ) -> Option<Disparity>
    where
        P: Fn(&DecisionRecord) -> bool,
        S: Fn(&DecisionRecord) -> bool,
    {
        let groups = self.by_value(axis);
        let mut kn: Vec<(String, u64, u64)> = Vec::new();
        let mut bools: BTreeMap<String, Vec<bool>> = BTreeMap::new();
        for (val, recs) in &groups {
            let eligible: Vec<&&DecisionRecord> = recs.iter().filter(|r| subset(r)).collect();
            if eligible.is_empty() {
                continue;
            }
            let outcomes: Vec<bool> = eligible.iter().map(|r| predicate(r)).collect();
            let k = outcomes.iter().filter(|&&x| x).count() as u64;
            let n = outcomes.len() as u64;
            kn.push((val.clone(), k, n));
            bools.insert(val.clone(), outcomes);
        }
        let gap = max_pairwise_gap(&kn)?;
        let a = &bools[&gap.group_a];
        let b = &bools[&gap.group_b];
        let p = permutation_pvalue(a, b, self.perm_iters, self.perm_seed);
        let n_total: usize = kn.iter().map(|(_, _, n)| *n as usize).sum();
        Some(Disparity {
            axis: axis.to_string(),
            metric,
            group_a: gap.group_a,
            group_b: gap.group_b,
            gap: gap.gap,
            ci_lo: gap.ci_lo,
            ci_hi: gap.ci_hi,
            p_value: p,
            n: n_total,
        })
    }

    /// Smallest per-group sample for `axis` (used by the no-peeking alert gate).
    fn min_group_n(&self, axis: &str) -> usize {
        self.by_value(axis)
            .values()
            .map(|v| v.len())
            .min()
            .unwrap_or(0)
    }

    /// Disparities that breach `bound` with p < `alpha`, but only once every group
    /// has at least `min_n` observations.
    pub fn alerts(&self, axis: &str, bound: f64, alpha: f64) -> Vec<Alert> {
        if self.min_group_n(axis) < self.min_n {
            return Vec::new();
        }
        self.disparities(axis)
            .into_iter()
            .filter(|d| d.gap >= bound && d.p_value < alpha)
            .map(|d| Alert { disparity: d, bound })
            .collect()
    }
}

impl SubgroupMonitor for StreamingMonitor {
    fn observe(&mut self, rec: &DecisionRecord) {
        self.records.push(rec.clone());
    }

    fn disparities(&self, axis: &str) -> Vec<Disparity> {
        let mut out = Vec::new();
        // Demographic parity: enforcement rate across all cases.
        if let Some(d) = self.disparity(
            axis,
            FairnessMetric::DemographicParity,
            |_| true,
            |r| r.action.is_enforced(),
        ) {
            out.push(d);
        }
        // Equalized odds: the larger of the TPR gap (among label==true) and the
        // FPR gap (among label==false).
        let tpr = self.disparity(
            axis,
            FairnessMetric::EqualizedOdds,
            |r| r.label,
            |r| r.action.is_enforced(),
        );
        let fpr = self.disparity(
            axis,
            FairnessMetric::EqualizedOdds,
            |r| !r.label,
            |r| r.action.is_enforced(),
        );
        match (tpr, fpr) {
            (Some(t), Some(f)) => out.push(if t.gap >= f.gap { t } else { f }),
            (Some(t), None) => out.push(t),
            (None, Some(f)) => out.push(f),
            (None, None) => {}
        }
        // Predictive parity: among enforced cases, the rate that were truly positive.
        if let Some(d) = self.disparity(
            axis,
            FairnessMetric::PredictiveParity,
            |r| r.action.is_enforced(),
            |r| r.label,
        ) {
            out.push(d);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::Action;

    fn rec(axis: &str, val: &str, enforced: bool, label: bool) -> DecisionRecord {
        let mut r = DecisionRecord::single(
            "c",
            axis,
            val,
            if enforced { Action::Deny } else { Action::Allow },
        );
        r.label = label;
        r
    }

    /// Stream a group with `k` enforced out of `n`, all label==true.
    fn feed(m: &mut StreamingMonitor, axis: &str, val: &str, k: usize, n: usize) {
        for i in 0..n {
            m.observe(&rec(axis, val, i < k, true));
        }
    }

    #[test]
    fn balanced_stream_has_no_significant_dpd() {
        let mut m = StreamingMonitor::new().with_perm(500, 1);
        feed(&mut m, "race", "White", 20, 40);
        feed(&mut m, "race", "Black", 20, 40);
        let dpd = m
            .disparities("race")
            .into_iter()
            .find(|d| d.metric == FairnessMetric::DemographicParity)
            .unwrap();
        assert!(dpd.gap < 0.05, "balanced gap should be ~0, got {}", dpd.gap);
        assert!(dpd.p_value > 0.2, "balanced p should be large, got {}", dpd.p_value);
    }

    #[test]
    fn planted_enforcement_gap_is_detected_with_correct_groups() {
        let mut m = StreamingMonitor::new().with_perm(1000, 7);
        feed(&mut m, "race", "Black", 32, 40); // 0.80 enforced
        feed(&mut m, "race", "White", 8, 40); //  0.20 enforced
        let dpd = m
            .disparities("race")
            .into_iter()
            .find(|d| d.metric == FairnessMetric::DemographicParity)
            .unwrap();
        assert_eq!(dpd.group_a, "Black");
        assert_eq!(dpd.group_b, "White");
        assert!((dpd.gap - 0.6).abs() < 1e-9);
        assert!(dpd.p_value < 0.05, "planted gap should be significant, got {}", dpd.p_value);
        assert!(dpd.ci_lo > 0.0, "CI should exclude zero");
    }

    #[test]
    fn no_alert_until_min_n_reached() {
        let mut m = StreamingMonitor::new().with_min_n(30).with_perm(500, 3);
        // Strong gap but only 10 per group — below min_n, so no alert yet.
        feed(&mut m, "race", "Black", 9, 10);
        feed(&mut m, "race", "White", 1, 10);
        assert!(m.alerts("race", 0.1, 0.05).is_empty(), "should not peek below min_n");
        // Now top up past min_n, preserving the gap.
        feed(&mut m, "race", "Black", 27, 30); // cumulative 36/40
        feed(&mut m, "race", "White", 3, 30); //  cumulative 4/40
        let alerts = m.alerts("race", 0.1, 0.05);
        // With all-positive labels the TPR (equalized-odds) component coincides
        // with demographic parity, so both metrics legitimately alert; assert the
        // gate opened and demographic parity is among them.
        assert!(!alerts.is_empty(), "should alert once min_n satisfied");
        assert!(alerts
            .iter()
            .any(|a| a.disparity.metric == FairnessMetric::DemographicParity));
    }

    #[test]
    fn predictive_parity_only_over_enforced_cases() {
        // Among enforced (predicted-positive) cases, what fraction were truly positive.
        let mut m = StreamingMonitor::new().with_perm(500, 1);
        // Black: 10 enforced, 9 of them truly positive (precision 0.9)
        for i in 0..10 {
            m.observe(&rec("race", "Black", true, i < 9));
        }
        // White: 10 enforced, 3 of them truly positive (precision 0.3)
        for i in 0..10 {
            m.observe(&rec("race", "White", true, i < 3));
        }
        let ppv = m
            .disparities("race")
            .into_iter()
            .find(|d| d.metric == FairnessMetric::PredictiveParity)
            .unwrap();
        assert_eq!(ppv.group_a, "Black");
        assert!((ppv.gap - 0.6).abs() < 1e-9);
    }
}
