//! equity::calibrator — equity-mitigation arms (E3 / H3). Each arm fits offline on
//! a labeled calibration split and ships as versioned, audited parameters; `apply`
//! re-decides one case from its score. Arms: `none` (global 0.5 threshold),
//! `group_threshold` (per-group threshold equalizing selection rate),
//! `reject_option` (defer near-boundary cases to review), `equalized_odds`
//! (per-group threshold equalizing the true-positive rate).

use std::collections::BTreeMap;

use super::{Action, DecisionRecord};

/// Which mitigation arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    None,
    GroupThreshold,
    RejectOption,
    EqualizedOdds,
}

/// Fitted, audited parameters for an arm over one axis.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalibratorParams {
    pub arm: Arm,
    pub axis: String,
    /// Per-group decision threshold on `score` (enforce when score >= threshold).
    pub thresholds: BTreeMap<String, f64>,
    /// Global threshold for groups unseen at fit time / arms without per-group ones.
    pub default_threshold: f64,
    /// Reject-option half-band: cases with |score - default_threshold| < band defer.
    pub reject_band: f64,
}

const Z: f64 = 1e-9;
const DEFAULT_THRESHOLD: f64 = 0.5;

pub struct EquityCalibrator;

/// Threshold on `scores` (ascending after sort) that enforces a `target` fraction
/// of them: pick the m-th largest so that ~`target·n` score at or above it.
fn quantile_threshold(scores: &[f64], target: f64) -> f64 {
    if scores.is_empty() {
        return DEFAULT_THRESHOLD;
    }
    let mut s = scores.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let m = (target * n as f64).round() as usize;
    if m == 0 {
        1.0 + Z // enforce nobody
    } else if m >= n {
        0.0 // enforce everybody
    } else {
        s[n - m] // the m-th largest score
    }
}

impl EquityCalibrator {
    /// Fit an arm on a calibration split for one protected axis.
    pub fn fit(arm: Arm, recs: &[DecisionRecord], axis: &str) -> CalibratorParams {
        let mut params = CalibratorParams {
            arm,
            axis: axis.to_string(),
            thresholds: BTreeMap::new(),
            default_threshold: DEFAULT_THRESHOLD,
            reject_band: 0.0,
        };
        // Group scored records by axis value.
        let mut groups: BTreeMap<String, Vec<&DecisionRecord>> = BTreeMap::new();
        for r in recs {
            if r.score.is_some() {
                if let Some(v) = r.value(axis) {
                    groups.entry(v.to_string()).or_default().push(r);
                }
            }
        }
        match arm {
            Arm::None => {}
            Arm::RejectOption => params.reject_band = 0.1,
            Arm::GroupThreshold => {
                // Target = overall enforcement rate at the global cut.
                let total = recs.iter().filter(|r| r.score.is_some()).count();
                let enforced = recs
                    .iter()
                    .filter(|r| r.score.map_or(false, |s| s >= DEFAULT_THRESHOLD))
                    .count();
                let target = if total == 0 {
                    0.0
                } else {
                    enforced as f64 / total as f64
                };
                for (val, gr) in &groups {
                    let scores: Vec<f64> = gr.iter().filter_map(|r| r.score).collect();
                    params
                        .thresholds
                        .insert(val.clone(), quantile_threshold(&scores, target));
                }
            }
            Arm::EqualizedOdds => {
                // Target true-positive rate at the global cut over truly-positive cases.
                let total_pos = recs.iter().filter(|r| r.label && r.score.is_some()).count();
                let enforced_pos = recs
                    .iter()
                    .filter(|r| r.label && r.score.map_or(false, |s| s >= DEFAULT_THRESHOLD))
                    .count();
                let target = if total_pos == 0 {
                    0.0
                } else {
                    enforced_pos as f64 / total_pos as f64
                };
                for (val, gr) in &groups {
                    let scores: Vec<f64> =
                        gr.iter().filter(|r| r.label).filter_map(|r| r.score).collect();
                    params
                        .thresholds
                        .insert(val.clone(), quantile_threshold(&scores, target));
                }
            }
        }
        params
    }

    /// Re-decide one case under fitted params. Records without a score fall back to
    /// their original action.
    pub fn apply(params: &CalibratorParams, rec: &DecisionRecord) -> Action {
        let score = match rec.score {
            Some(s) => s,
            None => return rec.action,
        };
        // Reject-option: boundary cases defer to human review.
        if params.reject_band > 0.0 && (score - params.default_threshold).abs() < params.reject_band {
            return Action::Escalate;
        }
        let threshold = rec
            .value(&params.axis)
            .and_then(|v| params.thresholds.get(v).copied())
            .unwrap_or(params.default_threshold);
        if score >= threshold {
            Action::Deny
        } else {
            Action::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(axis: &str, val: &str, score: f64, label: bool) -> DecisionRecord {
        let mut rec = DecisionRecord::single("c", axis, val, Action::Allow);
        rec.score = Some(score);
        rec.label = label;
        rec
    }

    /// Build a group's records from a list of (score, label).
    fn group(axis: &str, val: &str, items: &[(f64, bool)]) -> Vec<DecisionRecord> {
        items.iter().map(|&(s, l)| r(axis, val, s, l)).collect()
    }

    fn enforce_rate(params: &CalibratorParams, recs: &[DecisionRecord]) -> f64 {
        let k = recs
            .iter()
            .filter(|rec| EquityCalibrator::apply(params, rec).is_enforced())
            .count();
        k as f64 / recs.len() as f64
    }

    #[test]
    fn none_uses_global_half_threshold() {
        let p = EquityCalibrator::fit(Arm::None, &[], "race");
        assert_eq!(EquityCalibrator::apply(&p, &r("race", "X", 0.6, true)), Action::Deny);
        assert_eq!(EquityCalibrator::apply(&p, &r("race", "X", 0.4, true)), Action::Allow);
        assert_eq!(EquityCalibrator::apply(&p, &r("race", "X", 0.5, true)), Action::Deny);
    }

    #[test]
    fn missing_score_falls_back_to_original_action() {
        let p = EquityCalibrator::fit(Arm::None, &[], "race");
        let mut rec = DecisionRecord::single("c", "race", "X", Action::Escalate);
        rec.score = None;
        assert_eq!(EquityCalibrator::apply(&p, &rec), Action::Escalate);
    }

    #[test]
    fn group_threshold_shrinks_selection_rate_gap() {
        // Disadvantaged group "B" scores systematically higher → enforced more
        // under a global 0.5 cut. Group-threshold should equalize selection rates.
        let mut recs = Vec::new();
        // A: scores around 0.3–0.5 → ~20% over 0.5
        let a: Vec<(f64, bool)> = (0..10).map(|i| (0.25 + i as f64 * 0.03, i >= 8)).collect();
        // B: scores around 0.45–0.75 → ~80% over 0.5
        let b: Vec<(f64, bool)> = (0..10).map(|i| (0.45 + i as f64 * 0.03, i >= 8)).collect();
        recs.extend(group("race", "A", &a));
        recs.extend(group("race", "B", &b));

        let base = EquityCalibrator::fit(Arm::None, &recs, "race");
        let cal = EquityCalibrator::fit(Arm::GroupThreshold, &recs, "race");

        let ga: Vec<_> = group("race", "A", &a);
        let gb: Vec<_> = group("race", "B", &b);
        let base_gap = (enforce_rate(&base, &ga) - enforce_rate(&base, &gb)).abs();
        let cal_gap = (enforce_rate(&cal, &ga) - enforce_rate(&cal, &gb)).abs();
        assert!(
            cal_gap < base_gap,
            "group-threshold should shrink the gap: base={base_gap}, cal={cal_gap}"
        );
        assert!(cal_gap <= 0.2 + Z, "calibrated gap should be small, got {cal_gap}");
    }

    #[test]
    fn reject_option_defers_boundary_cases_to_review() {
        let p = EquityCalibrator::fit(Arm::RejectOption, &[], "race");
        assert_eq!(EquityCalibrator::apply(&p, &r("race", "X", 0.95, true)), Action::Deny);
        assert_eq!(EquityCalibrator::apply(&p, &r("race", "X", 0.05, false)), Action::Allow);
        // Near the 0.5 boundary → escalate to human review.
        assert_eq!(EquityCalibrator::apply(&p, &r("race", "X", 0.52, true)), Action::Escalate);
    }

    #[test]
    fn equalized_odds_equalizes_true_positive_rate() {
        // Among truly-positive cases, A and B should be enforced at similar rates.
        let a: Vec<(f64, bool)> = vec![(0.9, true), (0.8, true), (0.7, true), (0.6, true)];
        let b: Vec<(f64, bool)> = vec![(0.55, true), (0.45, true), (0.35, true), (0.25, true)];
        let mut recs = group("race", "A", &a);
        recs.extend(group("race", "B", &b));
        let cal = EquityCalibrator::fit(Arm::EqualizedOdds, &recs, "race");
        let tpr = |val: &str, items: &[(f64, bool)]| {
            let g = group("race", val, items);
            let pos: Vec<_> = g.iter().filter(|x| x.label).collect();
            let k = pos
                .iter()
                .filter(|x| EquityCalibrator::apply(&cal, x).is_enforced())
                .count();
            k as f64 / pos.len() as f64
        };
        let gap = (tpr("A", &a) - tpr("B", &b)).abs();
        assert!(gap <= 0.25 + Z, "equalized-odds TPR gap should be small, got {gap}");
    }
}
