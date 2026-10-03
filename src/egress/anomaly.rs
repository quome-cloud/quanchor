//! Per-session egress-volume anomaly monitor (paper 006, X4 trickle). Content-agnostic: accumulates
//! outbound bytes sent to EXTERNAL destinations across a session's calls and alarms when the running
//! total first exceeds a byte budget. Catches low-and-slow trickle that per-call thresholds miss;
//! internal-destination traffic (e.g. *@hospital.org) is not counted.

/// Monitors cumulative external-egress volume across a session's calls.
pub struct SessionMonitor {
    pub budget_bytes: usize,
    /// Destinations containing this substring are treated as internal (not counted).
    pub internal_domain: String,
}

impl SessionMonitor {
    pub fn new(budget_bytes: usize) -> Self {
        SessionMonitor { budget_bytes, internal_domain: "hospital.org".to_string() }
    }

    /// Each element of `calls` is one outbound call's (arg_path, value) pairs (including a "to" field).
    /// Returns the cumulative external byte count at the call where the running total first exceeds the
    /// budget (the "detection window"), or None if the session never exceeds it.
    pub fn evaluate(&self, calls: &[Vec<(String, String)>]) -> Option<usize> {
        let mut total = 0usize;
        for call in calls {
            let to = call.iter().find(|(k, _)| k == "to").map(|(_, v)| v.as_str()).unwrap_or("");
            let external = !to.is_empty() && !to.contains(&self.internal_domain);
            if !external {
                continue;
            }
            for (k, v) in call {
                if k != "to" {
                    total += v.len();
                }
            }
            if total > self.budget_bytes {
                return Some(total);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ext_calls(n: usize) -> Vec<Vec<(String, String)>> {
        (0..n).map(|i| vec![("to".to_string(), "ext@x.com".to_string()),
                            ("body".to_string(), format!("chunk {}", i % 10))]).collect()
    }
    #[test]
    fn trickle_to_external_alarms() {
        // 12 external calls (~7 bytes each) > budget 30 -> alarms.
        assert!(SessionMonitor::new(30).evaluate(&ext_calls(12)).is_some());
    }
    #[test]
    fn internal_traffic_never_alarms() {
        let calls: Vec<Vec<(String, String)>> = (0..12).map(|_| vec![
            ("to".to_string(), "pharmacy@hospital.org".to_string()),
            ("body".to_string(), "status update for bed 4".to_string())]).collect();
        assert!(SessionMonitor::new(30).evaluate(&calls).is_none());
    }
    #[test]
    fn single_small_external_call_under_budget() {
        assert!(SessionMonitor::new(30).evaluate(&ext_calls(1)).is_none());
    }
}
