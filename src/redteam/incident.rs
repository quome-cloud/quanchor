//! Incident-response engine (E3): detection over the 003 decision stream →
//! runbook actions, measuring mean-time-to-detect / contain / recover in stream
//! time-steps for {manual, automated} arms.
//!
//! Containment is the *modeled effect* of existing layers (isolate the agent,
//! revoke its 008 capability token, trip a circuit breaker, snapshot for
//! forensics) — live revocation wiring is out of scope for a reproducible
//! offline experiment (see the feature spec, §2).

use crate::redteam::{IncidentEvent, RunbookAction};
use serde::Serialize;

/// The fixed runbook fired on detection, in order. The first action contains
/// (IsolateAgent); the last completes recovery (Snapshot).
pub const RUNBOOK: [RunbookAction; 4] = [
    RunbookAction::IsolateAgent,
    RunbookAction::RevokeCapability,
    RunbookAction::TripBreaker,
    RunbookAction::Snapshot,
];

/// Per-incident response metrics, in stream time-steps. `-1.0` = not applicable
/// (e.g. no attack present, so nothing detected).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IncidentMetrics {
    pub mttd: f64,
    pub mttc: f64,
    pub mttr: f64,
    pub detected: bool,
    pub actions: Vec<RunbookAction>,
}

/// Respond to one incident stream.
///
/// * Detection fires at the first step whose `attack_signature` is `Some` AND
///   `score >= threshold`.
/// * MTTD = detect_step − attack_start_step (steps from first attack activity to
///   detection).
/// * Each runbook action takes `human_latency` steps in the manual arm (0 =
///   automated, immediate). Actions apply sequentially.
/// * MTTC = latency to the first (containing) action = `human_latency`.
/// * MTTR = latency to the last (recovery) action = `human_latency * (RUNBOOK.len()-1)`.
pub fn respond(events: &[IncidentEvent], threshold: f64, human_latency: u64) -> IncidentMetrics {
    let attack_start = events.iter().find(|e| e.attack_signature.is_some()).map(|e| e.step);
    let detect = events
        .iter()
        .find(|e| e.attack_signature.is_some() && e.score >= threshold)
        .map(|e| e.step);

    match (attack_start, detect) {
        (Some(start), Some(d)) => {
            let mttd = (d.saturating_sub(start)) as f64;
            let mttc = human_latency as f64;
            let mttr = (human_latency * (RUNBOOK.len() as u64 - 1)) as f64;
            IncidentMetrics {
                mttd,
                mttc,
                mttr,
                detected: true,
                actions: RUNBOOK.to_vec(),
            }
        }
        // Attack present but never crossed threshold, or no attack at all.
        _ => IncidentMetrics {
            mttd: -1.0,
            mttc: -1.0,
            mttr: -1.0,
            detected: false,
            actions: vec![],
        },
    }
}

/// Mean of the finite (>= 0) values in a slice; `-1.0` if none are finite.
pub fn mean_finite(xs: &[f64]) -> f64 {
    let v: Vec<f64> = xs.iter().copied().filter(|x| *x >= 0.0).collect();
    if v.is_empty() {
        -1.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(step: u64, sig: Option<&str>, score: f64) -> IncidentEvent {
        IncidentEvent {
            step,
            agent_id: "agent-1".into(),
            attack_signature: sig.map(|s| s.to_string()),
            score,
        }
    }

    fn attack_stream() -> Vec<IncidentEvent> {
        vec![
            ev(0, None, 0.1),
            ev(1, Some("inject"), 0.3), // attack starts, below threshold
            ev(2, Some("inject"), 0.9), // crosses threshold → detection
            ev(3, None, 0.1),
        ]
    }

    #[test]
    fn detection_fires_on_signature_above_threshold() {
        let m = respond(&attack_stream(), 0.5, 0);
        assert!(m.detected);
        assert_eq!(m.mttd, 1.0); // detect step 2 − attack start step 1
    }

    #[test]
    fn automated_mttc_less_than_manual() {
        let auto = respond(&attack_stream(), 0.5, 0);
        let manual = respond(&attack_stream(), 0.5, 5);
        assert!(auto.mttc < manual.mttc);
        assert_eq!(auto.mttc, 0.0);
        assert_eq!(manual.mttc, 5.0);
        assert!(auto.mttr < manual.mttr);
    }

    #[test]
    fn runbook_applies_all_four_actions() {
        let m = respond(&attack_stream(), 0.5, 0);
        assert_eq!(m.actions.len(), 4);
        assert_eq!(m.actions[0], RunbookAction::IsolateAgent);
        assert_eq!(m.actions[3], RunbookAction::Snapshot);
    }

    #[test]
    fn benign_stream_no_detection() {
        let benign = vec![ev(0, None, 0.1), ev(1, None, 0.2)];
        let m = respond(&benign, 0.5, 0);
        assert!(!m.detected);
        assert_eq!(m.mttd, -1.0);
        assert!(m.actions.is_empty());
    }

    #[test]
    fn mean_finite_ignores_sentinels() {
        assert_eq!(mean_finite(&[2.0, 4.0, -1.0]), 3.0);
        assert_eq!(mean_finite(&[-1.0, -1.0]), -1.0);
    }
}
