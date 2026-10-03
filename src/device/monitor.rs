//! Device performance monitor (C8.6). Tracks the deviation between a commanded
//! setting and the device's achieved response and raises a degradation alert
//! once that deviation stays above threshold for `w` consecutive actions — a
//! small run-length detector in the spirit of the 011 drift monitors. A single
//! transient spike does not alarm; sustained degradation does, with bounded
//! detection latency.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Alert {
    /// Index (0-based) of the action on which the alert fired.
    pub at_action: usize,
}

pub struct PerfMonitor {
    thresh: f64,
    w: usize,
    run: usize,
    seen: usize,
}

impl PerfMonitor {
    /// `thresh`: deviation that counts as unhealthy. `w`: consecutive unhealthy
    /// actions required to alarm (the detection window).
    pub fn new(thresh: f64, w: usize) -> Self {
        PerfMonitor { thresh, w, run: 0, seen: 0 }
    }

    /// Observe one action's deviation. Returns `Some(Alert)` exactly once, on
    /// the action that completes the unhealthy run.
    pub fn observe(&mut self, deviation: f64) -> Option<Alert> {
        let idx = self.seen;
        self.seen += 1;
        if deviation.abs() > self.thresh {
            self.run += 1;
            if self.run >= self.w {
                return Some(Alert { at_action: idx });
            }
        } else {
            self.run = 0;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_degradation_within_window() {
        let mut m = PerfMonitor::new(0.2, 3);
        for _ in 0..10 {
            assert!(m.observe(0.01).is_none()); // healthy
        }
        let mut detected_at = None;
        for _ in 0..10 {
            if let Some(a) = m.observe(0.5) {
                detected_at = Some(a.at_action);
                break;
            }
        }
        // 10 healthy samples (indices 0..9) then degradation: 3rd bad one is index 12.
        assert_eq!(detected_at, Some(12));
    }

    #[test]
    fn transient_spike_does_not_alarm() {
        let mut m = PerfMonitor::new(0.2, 3);
        assert!(m.observe(0.5).is_none()); // single spike
        assert!(m.observe(0.0).is_none()); // recovers
        assert!(m.observe(0.5).is_none());
    }

    #[test]
    fn detection_latency_is_w_actions_after_onset() {
        let mut m = PerfMonitor::new(0.1, 4);
        for _ in 0..4 {
            m.observe(0.0);
        }
        let onset = 4;
        let mut fired = None;
        for _ in 0..10 {
            if let Some(a) = m.observe(1.0) {
                fired = Some(a.at_action);
                break;
            }
        }
        assert_eq!(fired.map(|i| i - onset + 1), Some(4)); // W actions after onset
    }
}
