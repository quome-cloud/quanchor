//! Paper 011 — end-to-end monitor integration: feed a MonitorEvent stream through
//! a drift detector + the alert ladder and assert graduated alerts fire after a
//! labelled onset and stay silent on a clean stream.

use qfire::monitor::alert::AlertLadder;
use qfire::monitor::drift::Adwin;
use qfire::monitor::{AlertLevel, MonitorEvent, StreamDetector};

fn ev(case: u64, score: f64) -> MonitorEvent {
    MonitorEvent {
        case,
        ts: None,
        agent_id: "agentA".into(),
        tool: "read".into(),
        verdict: "allow".into(),
        score,
        outcome: Some(score < 0.5),
        autonomy_level: 1,
        autonomous: false,
    }
}

#[test]
fn onset_stream_raises_alert_after_t0() {
    let t0 = 200u64;
    let mut det = Adwin::new(0.05);
    let mut ladder = AlertLadder::default();
    let mut first_alert = None;
    for i in 0..400u64 {
        let score = if i < t0 { 0.1 } else { 0.7 };
        if let Some(sig) = det.observe(i, ev(i, score).score) {
            let a = ladder.classify(&sig);
            if first_alert.is_none() {
                first_alert = Some((a.case, a.level));
            }
        }
    }
    let (case, level) = first_alert.expect("an onset stream must raise an alert");
    assert!(case >= t0, "alert fires after onset, got {case}");
    assert!(level >= AlertLevel::Notify, "a clear shift should be at least Notify");
}

#[test]
fn clean_stream_is_silent() {
    let mut det = Adwin::new(0.002);
    let mut ladder = AlertLadder::default();
    for i in 0..400u64 {
        let score = 0.3 + ((i as f64 * 0.7).sin()) * 0.02;
        if let Some(sig) = det.observe(i, ev(i, score).score) {
            ladder.classify(&sig);
        }
    }
    assert_eq!(ladder.clinician_volume(), 0, "a clean stream must not page a clinician");
}
