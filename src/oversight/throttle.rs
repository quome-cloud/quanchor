//! Alarm-fatigue mechanism: as outstanding human prompts pile up, raise the
//! auto-allow threshold so only higher-risk actions still escalate. Linear in
//! outstanding count, capped.

#[derive(Debug, Clone, Copy)]
pub struct LoadThrottle {
    pub per_ticket: f64,
    pub cap: f64,
}

impl Default for LoadThrottle {
    fn default() -> Self {
        LoadThrottle {
            per_ticket: 0.02,
            cap: 0.30,
        }
    }
}

impl LoadThrottle {
    /// How much to raise `allow_below` given `outstanding` parked prompts.
    pub fn allow_shift(&self, outstanding: u32) -> f64 {
        (self.per_ticket * outstanding as f64).min(self.cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_raises_allow_threshold_under_load() {
        let th = LoadThrottle {
            per_ticket: 0.02,
            cap: 0.30,
        };
        assert_eq!(th.allow_shift(0), 0.0);
        assert!((th.allow_shift(5) - 0.10).abs() < 1e-9);
        assert_eq!(th.allow_shift(100), 0.30); // capped
    }
}
