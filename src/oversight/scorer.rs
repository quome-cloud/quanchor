//! Calibrated risk scorer: a weighted logit over the signals, squashed by a
//! temperature-scaled sigmoid. Weights + temperature are fit offline on
//! TriageBench and shipped as versioned, audited config.

use crate::oversight::signals::OversightSignals;
use serde::{Deserialize, Serialize};

/// Versioned calibration parameters (shipped as config; audited per 002/003).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub version: String,
    pub bias: f64,
    pub w_block_conf: f64,
    pub w_taint: f64,
    pub w_policy: f64,
    pub w_provenance: f64,
    pub w_identity: f64,
    /// Multiplied by `tool_risk` index / 3.0.
    pub w_tool_risk: f64,
    pub temperature: f64,
}

impl Default for Calibration {
    /// The fitted operating point, machine-fit on TriageBench by
    /// `scripts/009-hitl-calibration/fit_calibration.py` and mirrored in
    /// `datasets/009-hitl-calibration/calibration_fitted.json` (a Rust test
    /// asserts they stay in sync). Note `w_tool_risk` ≈ 0: the scorer estimates
    /// P(harm) from signals; tool-risk caution lives in the router bands, not
    /// the score.
    fn default() -> Self {
        Calibration {
            version: "009-fitted-v1".into(),
            bias: -3.668004,
            w_block_conf: 9.845944,
            w_taint: 1.980851,
            w_policy: -3.040942,
            w_provenance: 0.805425,
            w_identity: 0.789926,
            w_tool_risk: -0.25514,
            temperature: 0.75,
        }
    }
}

pub struct RiskScorer {
    cal: Calibration,
}

impl RiskScorer {
    pub fn new(cal: Calibration) -> Self {
        RiskScorer { cal }
    }

    pub fn calibration(&self) -> &Calibration {
        &self.cal
    }

    /// Risk in [0,1]. `None` signals contribute 0 (absent ⇒ no evidence).
    pub fn score(&self, s: &OversightSignals) -> f64 {
        let c = &self.cal;
        let tool_idx = s.tool_risk as i32 as f64 / 3.0;
        let logit = c.bias
            + c.w_block_conf * s.max_block_confidence
            + c.w_taint * s.taint_level.unwrap_or(0.0)
            + c.w_policy * s.policy_margin.unwrap_or(0.0)
            + c.w_provenance * s.provenance_trust.unwrap_or(0.0)
            + c.w_identity * s.identity_trust.unwrap_or(0.0)
            + c.w_tool_risk * tool_idx;
        let t = c.temperature.max(1e-6);
        1.0 / (1.0 + (-(logit / t)).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oversight::signals::{OversightSignals, ToolRisk};

    #[test]
    fn risk_is_bounded_and_deterministic() {
        let cal = Calibration::default();
        let s = OversightSignals::for_tool(ToolRisk::Low);
        let r1 = RiskScorer::new(cal.clone()).score(&s);
        let r2 = RiskScorer::new(cal).score(&s);
        assert!((0.0..=1.0).contains(&r1));
        assert_eq!(r1, r2); // deterministic
    }

    #[test]
    fn more_block_signal_raises_risk_monotonically() {
        let cal = Calibration::default();
        let scorer = RiskScorer::new(cal);
        let clean = OversightSignals::for_tool(ToolRisk::High);
        let mut tainted = clean.clone();
        tainted.taint_level = Some(1.0);
        tainted.max_block_confidence = 1.0;
        assert!(scorer.score(&tainted) > scorer.score(&clean));
    }

    #[test]
    fn default_calibration_matches_fitted_config() {
        // The shipped default() must equal the machine-fit params committed at
        // datasets/009-hitl-calibration/calibration_fitted.json (no silent drift).
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/datasets/009-hitl-calibration/calibration_fitted.json"
        );
        let json = std::fs::read_to_string(path).expect("fitted calibration config present");
        let fitted: Calibration = serde_json::from_str(&json).expect("deserialize Calibration");
        let d = Calibration::default();
        assert_eq!(fitted.version, d.version);
        assert_eq!(fitted.bias, d.bias);
        assert_eq!(fitted.w_block_conf, d.w_block_conf);
        assert_eq!(fitted.w_taint, d.w_taint);
        assert_eq!(fitted.w_policy, d.w_policy);
        assert_eq!(fitted.w_provenance, d.w_provenance);
        assert_eq!(fitted.w_identity, d.w_identity);
        assert_eq!(fitted.w_tool_risk, d.w_tool_risk);
        assert_eq!(fitted.temperature, d.temperature);
    }

    #[test]
    fn temperature_above_one_softens_extremes() {
        let mut hot = Calibration::default();
        hot.temperature = 1.0;
        let mut cool = Calibration::default();
        cool.temperature = 3.0;
        let mut s = OversightSignals::for_tool(ToolRisk::High);
        s.taint_level = Some(1.0);
        s.max_block_confidence = 1.0;
        let r_hot = RiskScorer::new(hot).score(&s);
        let r_cool = RiskScorer::new(cool).score(&s);
        assert!(r_cool < r_hot); // higher T pulls toward 0.5
    }
}
