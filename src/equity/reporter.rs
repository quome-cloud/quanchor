//! equity::reporter — the C7.8 regulatory equity report. Serializes per-axis
//! disparities (each metric, with CI + p-value), the alerts that fired, and any
//! mitigation deltas into a deterministic JSON document for the audit log.

use super::monitor::{Alert, StreamingMonitor};
use super::{Disparity, SubgroupMonitor};

/// A C7.8 equity report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EquityReport {
    pub qfire_version: String,
    /// Per-axis disparities under every metric.
    pub axes: Vec<AxisReport>,
    /// Total alerts across all axes.
    pub alert_count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AxisReport {
    pub axis: String,
    pub disparities: Vec<Disparity>,
    pub alerts: Vec<Alert>,
}

impl EquityReport {
    /// Build the report from a monitor, alerting on `bound`/`alpha` per axis.
    pub fn from_monitor(monitor: &StreamingMonitor, bound: f64, alpha: f64) -> EquityReport {
        let mut axes = Vec::new();
        let mut alert_count = 0;
        for axis in monitor.axes() {
            let disparities = monitor.disparities(&axis);
            let alerts = monitor.alerts(&axis, bound, alpha);
            alert_count += alerts.len();
            axes.push(AxisReport {
                axis,
                disparities,
                alerts,
            });
        }
        EquityReport {
            qfire_version: crate::VERSION.to_string(),
            axes,
            alert_count,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("equity report serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::{Action, DecisionRecord, SubgroupMonitor};

    fn rec(val: &str, enforced: bool) -> DecisionRecord {
        let mut r = DecisionRecord::single(
            "c",
            "race",
            val,
            if enforced { Action::Deny } else { Action::Allow },
        );
        r.label = true;
        r
    }

    #[test]
    fn report_captures_axis_and_alert() {
        let mut m = StreamingMonitor::new().with_min_n(10).with_perm(500, 1);
        for i in 0..40 {
            m.observe(&rec("Black", i < 32)); // 0.80
        }
        for i in 0..40 {
            m.observe(&rec("White", i < 8)); // 0.20
        }
        let report = EquityReport::from_monitor(&m, 0.1, 0.05);
        assert_eq!(report.axes.len(), 1);
        assert_eq!(report.axes[0].axis, "race");
        assert!(report.alert_count >= 1, "a strong disparity should alert");

        let json = report.to_json();
        assert!(json.get("axes").is_some());
        assert!(json.get("alert_count").is_some());
        assert_eq!(json["axes"][0]["axis"], "race");
    }

    #[test]
    fn report_is_deterministic() {
        let mut m = StreamingMonitor::new().with_perm(300, 5);
        for i in 0..30 {
            m.observe(&rec("A", i < 15));
            m.observe(&rec("B", i < 15));
        }
        let j1 = EquityReport::from_monitor(&m, 0.1, 0.05).to_json();
        let j2 = EquityReport::from_monitor(&m, 0.1, 0.05).to_json();
        assert_eq!(j1, j2);
    }
}
