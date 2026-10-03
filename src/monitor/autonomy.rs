//! Autonomy meter — flags autonomy creep (D2) against an authorized envelope.
//!
//! The [`AutonomyEnvelope`] models the surface the 012 regulatory passport will
//! later supply (`max_autonomous_risk_tier`, `max_autonomous_fraction`). The
//! [`AutonomyMeter`] tracks live autonomy KPIs over a sliding window and emits a
//! creep [`DriftSignal`] when the agent acts outside the envelope.

use super::DriftSignal;
use std::collections::VecDeque;

/// The authorized-autonomy envelope (models the 012 passport contract).
#[derive(Debug, Clone)]
pub struct AutonomyEnvelope {
    /// Highest risk tier the agent may actuate autonomously.
    pub max_autonomous_risk_tier: u8,
    /// Maximum fraction of recent actions taken autonomously.
    pub max_autonomous_fraction: f64,
    /// Sliding-window size for the fraction KPI.
    pub window: usize,
}

/// Tracks live autonomy KPIs over a sliding window and flags creep past the envelope.
pub struct AutonomyMeter {
    env: AutonomyEnvelope,
    recent: VecDeque<bool>,
    auto_count: usize,
}

impl AutonomyMeter {
    pub fn new(env: AutonomyEnvelope) -> Self {
        Self { env, recent: VecDeque::new(), auto_count: 0 }
    }

    /// Observe one action; returns a creep signal if it breaches the envelope.
    pub fn observe(&mut self, case: u64, autonomous: bool, level: u8) -> Option<DriftSignal> {
        self.recent.push_back(autonomous);
        if autonomous {
            self.auto_count += 1;
        }
        if self.recent.len() > self.env.window {
            if let Some(true) = self.recent.pop_front() {
                self.auto_count -= 1;
            }
        }
        // Tier violation: an autonomous action above the authorized tier — immediate.
        if autonomous && level > self.env.max_autonomous_risk_tier {
            return Some(DriftSignal {
                detector: "autonomy_tier".into(),
                case,
                statistic: level as f64,
                severity: 1.0,
            });
        }
        // Fraction violation: sustained autonomous share beyond the envelope.
        if self.recent.len() >= self.env.window {
            let frac = self.auto_count as f64 / self.recent.len() as f64;
            if frac > self.env.max_autonomous_fraction {
                let head = (1.0 - self.env.max_autonomous_fraction).max(1e-6);
                let severity = ((frac - self.env.max_autonomous_fraction) / head).min(1.0);
                return Some(DriftSignal {
                    detector: "autonomy_frac".into(),
                    case,
                    statistic: frac,
                    severity,
                });
            }
        }
        None
    }

    pub fn reset(&mut self) {
        self.recent.clear();
        self.auto_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_tier_violation_immediately() {
        let env = AutonomyEnvelope { max_autonomous_risk_tier: 2, max_autonomous_fraction: 0.9, window: 50 };
        let mut m = AutonomyMeter::new(env);
        assert!(m.observe(0, true, 4).is_some());
    }

    #[test]
    fn flags_fraction_creep() {
        let env = AutonomyEnvelope { max_autonomous_risk_tier: 5, max_autonomous_fraction: 0.5, window: 20 };
        let mut m = AutonomyMeter::new(env);
        let mut fired = false;
        for i in 0..40u64 {
            if m.observe(i, true, 1).is_some() {
                fired = true;
            }
        }
        assert!(fired, "sustained high autonomous fraction must flag creep");
    }

    #[test]
    fn silent_within_envelope() {
        let env = AutonomyEnvelope { max_autonomous_risk_tier: 5, max_autonomous_fraction: 0.5, window: 20 };
        let mut m = AutonomyMeter::new(env);
        let mut alarms = 0;
        for i in 0..100u64 {
            let auto = i % 4 == 0;
            if m.observe(i, auto, 1).is_some() {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0);
    }
}
