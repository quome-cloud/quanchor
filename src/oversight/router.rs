//! Pure routing: risk × tool-risk-class × load → {auto-allow, auto-deny,
//! escalate}. No I/O, no ticket minting (that's `channel`). Thresholds are
//! config; the paper ships these defaults.

use crate::oversight::signals::ToolRisk;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteClass {
    AutoAllow,
    AutoDeny,
    Escalate,
}

/// Current escalation load (outstanding human prompts). Drives the throttle.
#[derive(Debug, Clone, Copy, Default)]
pub struct Load {
    pub outstanding: u32,
}

/// Per-tool-risk escalation band: below `allow_below` ⇒ auto-allow, above
/// `deny_above` ⇒ auto-deny, in between ⇒ escalate. Higher risk classes have a
/// lower `allow_below` (escalate earlier).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Band {
    pub allow_below: f64,
    pub deny_above: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    pub low: Band,
    pub medium: Band,
    pub high: Band,
    pub life_critical: Band,
}

impl Default for Thresholds {
    /// Recommended escalation bands (P(harm) thresholds), shipped alongside the
    /// fitted calibration in `datasets/009-hitl-calibration/calibration_fitted.json`.
    /// Higher tool-risk classes escalate a wider middle (more caution).
    fn default() -> Self {
        Thresholds {
            low: Band {
                allow_below: 0.25,
                deny_above: 0.80,
            },
            medium: Band {
                allow_below: 0.18,
                deny_above: 0.85,
            },
            high: Band {
                allow_below: 0.12,
                deny_above: 0.90,
            },
            life_critical: Band {
                allow_below: 0.06,
                deny_above: 0.95,
            },
        }
    }
}

impl Thresholds {
    pub fn band(&self, c: ToolRisk) -> &Band {
        match c {
            ToolRisk::Low => &self.low,
            ToolRisk::Medium => &self.medium,
            ToolRisk::High => &self.high,
            ToolRisk::LifeCritical => &self.life_critical,
        }
    }
}

/// Route one action. `Load` is accepted here so a throttle can pre-adjust the
/// effective risk; the base table itself ignores load (throttle is applied by
/// the guard before calling).
pub fn route(risk: f64, class: ToolRisk, _load: Load, th: &Thresholds) -> RouteClass {
    let b = th.band(class);
    if risk < b.allow_below {
        RouteClass::AutoAllow
    } else if risk > b.deny_above {
        RouteClass::AutoDeny
    } else {
        RouteClass::Escalate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oversight::signals::ToolRisk;

    #[test]
    fn low_risk_band_auto_allows_then_escalates_then_denies() {
        let th = Thresholds::default();
        assert_eq!(
            route(0.05, ToolRisk::Low, Load::default(), &th),
            RouteClass::AutoAllow
        );
        assert_eq!(
            route(0.50, ToolRisk::Low, Load::default(), &th),
            RouteClass::Escalate
        );
        assert_eq!(
            route(0.99, ToolRisk::Low, Load::default(), &th),
            RouteClass::AutoDeny
        );
    }

    #[test]
    fn life_critical_escalates_earlier_than_low() {
        let th = Thresholds::default();
        assert_eq!(
            route(0.20, ToolRisk::Low, Load::default(), &th),
            RouteClass::AutoAllow
        );
        assert_eq!(
            route(0.20, ToolRisk::LifeCritical, Load::default(), &th),
            RouteClass::Escalate
        );
    }
}
