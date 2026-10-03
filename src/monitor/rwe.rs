//! Real-world-evidence (RWE) exporter — aggregates monitored evidence into a
//! passport-ready bundle (links 010/012) and reproduces the injected effect size
//! as the RWE validity check.

use serde::{Deserialize, Serialize};

/// Passport-ready evidence bundle for one detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RweEvidence {
    pub detector: String,
    pub episodes_detected: u64,
    pub mean_detection_latency: f64,
    pub far: f64,
    /// Reproduced injected effect size (RWE validity check).
    pub estimated_effect: f64,
}

/// Accumulates monitored evidence and reproduces the injected effect size.
pub struct RweExporter {
    detector: String,
    latencies: Vec<f64>,
    effects: Vec<f64>,
    clean_streams: u64,
    false_alarms: u64,
}

impl RweExporter {
    pub fn new(detector: &str) -> Self {
        Self {
            detector: detector.to_string(),
            latencies: Vec::new(),
            effects: Vec::new(),
            clean_streams: 0,
            false_alarms: 0,
        }
    }

    pub fn record_episode(&mut self, latency: u64, observed_effect: f64) {
        self.latencies.push(latency as f64);
        self.effects.push(observed_effect);
    }

    pub fn record_clean_stream(&mut self, false_alarm: bool) {
        self.clean_streams += 1;
        if false_alarm {
            self.false_alarms += 1;
        }
    }

    pub fn finish(&self) -> RweEvidence {
        let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        RweEvidence {
            detector: self.detector.clone(),
            episodes_detected: self.latencies.len() as u64,
            mean_detection_latency: mean(&self.latencies),
            far: if self.clean_streams == 0 { 0.0 } else { self.false_alarms as f64 / self.clean_streams as f64 },
            estimated_effect: mean(&self.effects),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_and_serializes() {
        let mut e = RweExporter::new("cusum");
        e.record_episode(12, 0.4);
        e.record_episode(20, 0.6);
        e.record_clean_stream(false);
        e.record_clean_stream(true);
        let ev = e.finish();
        assert_eq!(ev.episodes_detected, 2);
        assert!((ev.mean_detection_latency - 16.0).abs() < 1e-9);
        assert!((ev.far - 0.5).abs() < 1e-9);
        assert!((ev.estimated_effect - 0.5).abs() < 1e-9);
        let j = serde_json::to_string(&ev).unwrap();
        assert!(j.contains("\"detector\":\"cusum\""));
    }
}
