//! Graduated alert ladder (models the 009 oversight-router contract).
//!
//! [`AlertLadder`] maps a [`DriftSignal`]'s severity to a graduated
//! [`AlertLevel`] (`Log → Notify → Throttle → Halt`) and tallies the per-level
//! volume so H4 (alert-fatigue) can account for clinician-facing load.

use super::{Alert, AlertLevel, DriftSignal};
use std::collections::BTreeMap;

/// Maps detector-signal severity to a graduated response level and tallies volume.
/// Thresholds are inclusive lower bounds.
pub struct AlertLadder {
    notify: f64,
    throttle: f64,
    halt: f64,
    counts: BTreeMap<AlertLevel, u64>,
}

impl Default for AlertLadder {
    fn default() -> Self {
        Self { notify: 0.2, throttle: 0.5, halt: 0.85, counts: BTreeMap::new() }
    }
}

impl AlertLadder {
    pub fn new(notify: f64, throttle: f64, halt: f64) -> Self {
        Self { notify, throttle, halt, counts: BTreeMap::new() }
    }

    pub fn classify(&mut self, sig: &DriftSignal) -> Alert {
        let level = if sig.severity >= self.halt {
            AlertLevel::Halt
        } else if sig.severity >= self.throttle {
            AlertLevel::Throttle
        } else if sig.severity >= self.notify {
            AlertLevel::Notify
        } else {
            AlertLevel::Log
        };
        *self.counts.entry(level).or_insert(0) += 1;
        Alert {
            level,
            case: sig.case,
            source: sig.detector.clone(),
            detail: format!("stat={:.4}", sig.statistic),
        }
    }

    pub fn count(&self, level: AlertLevel) -> u64 {
        self.counts.get(&level).copied().unwrap_or(0)
    }

    /// Clinician-facing volume = Notify + Throttle + Halt (the H4 budget).
    pub fn clinician_volume(&self) -> u64 {
        self.count(AlertLevel::Notify) + self.count(AlertLevel::Throttle) + self.count(AlertLevel::Halt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{AlertLevel, DriftSignal};

    fn sig(sev: f64) -> DriftSignal {
        DriftSignal { detector: "d".into(), case: 1, statistic: 0.0, severity: sev }
    }

    #[test]
    fn maps_severity_to_levels() {
        let mut l = AlertLadder::default();
        assert_eq!(l.classify(&sig(0.05)).level, AlertLevel::Log);
        assert_eq!(l.classify(&sig(0.3)).level, AlertLevel::Notify);
        assert_eq!(l.classify(&sig(0.6)).level, AlertLevel::Throttle);
        assert_eq!(l.classify(&sig(0.95)).level, AlertLevel::Halt);
    }

    #[test]
    fn counts_accumulate() {
        let mut l = AlertLadder::default();
        l.classify(&sig(0.3));
        l.classify(&sig(0.3));
        assert_eq!(l.count(AlertLevel::Notify), 2);
        assert_eq!(l.clinician_volume(), 2);
    }
}
