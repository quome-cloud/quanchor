//! E1–E5 scorer for paper 011: replay HAARF-Drift streams through the `monitor`
//! detectors and emit per-experiment JSON for the Python summarizer.
//!
//! Off-hot-path: detectors consume the read-side `MonitorEvent` stream; no proxy
//! or network is involved. Deterministic.
//!
//! Run:
//!   cargo run --release --bin driftbench -- \
//!     --data datasets/011-drift-monitoring --out results/011-drift-monitoring --exp all

use qfire::monitor::autonomy::{AutonomyEnvelope, AutonomyMeter};
use qfire::monitor::behavior::BehaviorBaseline;
use qfire::monitor::drift::{Adwin, Conformal, Cusum, DistDistance, DistMetric};
use qfire::monitor::{MonitorEvent, StreamDetector};
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct Label {
    file: String,
    family: String,
    kind: String,
    t0: Option<u64>,
    delta: String,
    #[allow(dead_code)]
    onset_magnitude: f64,
}

const DRIFT_DETECTORS: [&str; 4] = ["cusum", "adwin", "conformal", "distdistance"];

fn default_thresh(name: &str) -> f64 {
    match name {
        "cusum" => 0.5,
        "adwin" => 0.05,
        "conformal" => 0.01,
        "distdistance" => 0.2,
        _ => unreachable!(),
    }
}

/// ROC threshold grid (ordered least→most sensitive where possible).
fn roc_grid(name: &str) -> Vec<f64> {
    match name {
        "cusum" => vec![3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.2],
        "adwin" => vec![0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3],
        "conformal" => vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        "distdistance" => vec![0.5, 0.35, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02],
        _ => unreachable!(),
    }
}

fn make_detector(name: &str, thresh: f64) -> Box<dyn StreamDetector> {
    match name {
        "cusum" => Box::new(Cusum::new(0.05, thresh, 100)),
        "adwin" => Box::new(Adwin::new(thresh)),
        "conformal" => Box::new(Conformal::new(100, 0.5, thresh)),
        "distdistance" => Box::new(DistDistance::new(200, 100, 10, thresh, DistMetric::Psi)),
        _ => unreachable!(),
    }
}

fn load_labels(data: &str) -> anyhow::Result<Vec<Label>> {
    let text = std::fs::read_to_string(Path::new(data).join("labels.jsonl"))?;
    let mut v = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        v.push(serde_json::from_str(line)?);
    }
    Ok(v)
}

fn load_stream(data: &str, file: &str) -> anyhow::Result<Vec<MonitorEvent>> {
    let text = std::fs::read_to_string(Path::new(data).join(file))?;
    let mut v = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        v.push(serde_json::from_str(line)?);
    }
    Ok(v)
}

/// Run a drift detector over a stream's `score`. Returns
/// (latency past onset if detected, any_alarm_anywhere).
fn eval_drift(events: &[MonitorEvent], t0: Option<u64>, det: &mut dyn StreamDetector) -> (Option<u64>, bool) {
    let mut latency = None;
    let mut any = false;
    for ev in events {
        if det.observe(ev.case, ev.score).is_some() {
            any = true;
            if let (Some(t0), None) = (t0, latency) {
                if ev.case >= t0 {
                    latency = Some(ev.case - t0);
                }
            }
        }
    }
    (latency, any)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

// ── E1: drift detection latency vs magnitude + ROC ───────────────────────────
fn exp_e1(data: &str, labels: &[Label]) -> anyhow::Result<serde_json::Value> {
    let onset: Vec<&Label> = labels.iter().filter(|l| l.kind == "drift" && l.t0.is_some()).collect();
    let clean: Vec<&Label> = labels.iter().filter(|l| l.kind == "drift" && l.t0.is_none()).collect();

    // Cells: per (detector, family, delta) — detection rate + mean latency at the operating point.
    let mut cells = Vec::new();
    let mut families: Vec<String> = onset.iter().map(|l| l.family.clone()).collect();
    families.sort();
    families.dedup();
    let mut deltas: Vec<String> = onset.iter().map(|l| l.delta.clone()).collect();
    deltas.sort();
    deltas.dedup();

    for det_name in DRIFT_DETECTORS {
        for fam in &families {
            for delta in &deltas {
                let cell: Vec<&&Label> = onset.iter().filter(|l| &l.family == fam && &l.delta == delta).collect();
                if cell.is_empty() { continue; }
                let mut latencies = Vec::new();
                let mut detected = 0u64;
                let mut false_pre = 0u64;
                for l in &cell {
                    let events = load_stream(data, &l.file)?;
                    let mut det = make_detector(det_name, default_thresh(det_name));
                    let (lat, any) = eval_drift(&events, l.t0, det.as_mut());
                    if let Some(x) = lat { detected += 1; latencies.push(x as f64); }
                    else if any { false_pre += 1; }
                }
                cells.push(json!({
                    "detector": det_name, "family": fam, "delta": delta,
                    "n": cell.len(), "detected": detected,
                    "detection_rate": detected as f64 / cell.len() as f64,
                    "mean_latency": mean(&latencies),
                    "latencies": latencies,
                    "false_pre": false_pre,
                }));
            }
        }
    }

    // ROC: TPR over all onset streams vs FPR over clean streams, sweeping threshold.
    let mut roc = Vec::new();
    for det_name in DRIFT_DETECTORS {
        let mut points = Vec::new();
        for &thresh in &roc_grid(det_name) {
            let mut tp = 0u64;
            let mut lat_sum = Vec::new();
            for l in &onset {
                let events = load_stream(data, &l.file)?;
                let mut det = make_detector(det_name, thresh);
                let (lat, _) = eval_drift(&events, l.t0, det.as_mut());
                if let Some(x) = lat { tp += 1; lat_sum.push(x as f64); }
            }
            let mut fp = 0u64;
            for l in &clean {
                let events = load_stream(data, &l.file)?;
                let mut det = make_detector(det_name, thresh);
                let (_, any) = eval_drift(&events, None, det.as_mut());
                if any { fp += 1; }
            }
            points.push(json!({
                "thresh": thresh,
                "tpr": tp as f64 / onset.len().max(1) as f64,
                "fpr": fp as f64 / clean.len().max(1) as f64,
                "mean_latency": mean(&lat_sum),
            }));
        }
        roc.push(json!({ "detector": det_name, "points": points }));
    }

    Ok(json!({ "cells": cells, "roc": roc,
               "operating_thresholds": DRIFT_DETECTORS.iter().map(|d| json!({"detector": d, "thresh": default_thresh(d)})).collect::<Vec<_>>(),
               "n_onset": onset.len(), "n_clean": clean.len() }))
}

// ── E2: delayed / partial labels (D4) ────────────────────────────────────────
/// Deterministic per-case labeling: keep a `frac` subsample, deliver delayed by `delay`.
fn delayed_signal(events: &[MonitorEvent], delay: u64, frac: f64) -> Vec<(u64, f64)> {
    let cutoff = (frac * 1000.0) as u64;
    let mut arrivals: Vec<(u64, f64)> = Vec::new();
    for ev in events {
        let h = ev.case.wrapping_mul(2_654_435_761) % 1000;
        if h < cutoff {
            let fail = if matches!(ev.outcome, Some(false)) { 1.0 } else { 0.0 };
            arrivals.push((ev.case + delay, fail));
        }
    }
    arrivals.sort_by_key(|(c, _)| *c);
    arrivals
}

fn exp_e2(data: &str, labels: &[Label]) -> anyhow::Result<serde_json::Value> {
    let onset: Vec<&Label> = labels.iter().filter(|l| l.kind == "drift" && l.t0.is_some()).collect();
    let clean: Vec<&Label> = labels.iter().filter(|l| l.kind == "drift" && l.t0.is_none()).collect();
    let delays = [0u64, 50, 150];
    let fracs = [1.0f64, 0.5, 0.2];
    let mut grid = Vec::new();
    for &delay in &delays {
        for &frac in &fracs {
            let mut latencies = Vec::new();
            let mut detected = 0u64;
            for l in &onset {
                let events = load_stream(data, &l.file)?;
                let arr = delayed_signal(&events, delay, frac);
                // CUSUM on the binary failure-arrival signal (mean ~ base failure rate).
                let mut det = Cusum::new(0.25, 4.0, 80);
                let mut lat = None;
                for (case, fail) in &arr {
                    if det.observe(*case, *fail).is_some() {
                        if let (Some(t0), None) = (l.t0, lat) {
                            if *case >= t0 { lat = Some(*case - t0); }
                        }
                    }
                }
                if let Some(x) = lat { detected += 1; latencies.push(x as f64); }
            }
            let mut fp = 0u64;
            for l in &clean {
                let events = load_stream(data, &l.file)?;
                let arr = delayed_signal(&events, delay, frac);
                let mut det = Cusum::new(0.25, 4.0, 80);
                let mut any = false;
                for (case, fail) in &arr {
                    if det.observe(*case, *fail).is_some() { any = true; }
                }
                if any { fp += 1; }
            }
            grid.push(json!({
                "delay": delay, "frac": frac, "detector": "cusum",
                "detection_rate": detected as f64 / onset.len().max(1) as f64,
                "mean_latency": mean(&latencies),
                "far": fp as f64 / clean.len().max(1) as f64,
            }));
        }
    }
    Ok(json!({ "grid": grid, "n_onset": onset.len(), "n_clean": clean.len() }))
}

// ── E3: autonomy creep vs envelope ───────────────────────────────────────────
fn exp_e3(data: &str, labels: &[Label]) -> anyhow::Result<serde_json::Value> {
    let creep: Vec<&Label> = labels.iter().filter(|l| l.kind == "autonomy" && l.t0.is_some()).collect();
    let clean: Vec<&Label> = labels.iter().filter(|l| l.kind == "autonomy" && l.t0.is_none()).collect();
    let max_fracs = [0.7f64, 0.6, 0.5, 0.4];
    let mut envelopes = Vec::new();
    for &mf in &max_fracs {
        let mut latencies = Vec::new();
        let mut detected = 0u64;
        for l in &creep {
            let events = load_stream(data, &l.file)?;
            let env = AutonomyEnvelope { max_autonomous_risk_tier: 2, max_autonomous_fraction: mf, window: 100 };
            let mut m = AutonomyMeter::new(env);
            let mut lat = None;
            for ev in &events {
                if m.observe(ev.case, ev.autonomous, ev.autonomy_level).is_some() {
                    if let (Some(t0), None) = (l.t0, lat) {
                        if ev.case >= t0 { lat = Some(ev.case - t0); }
                    }
                }
            }
            if let Some(x) = lat { detected += 1; latencies.push(x as f64); }
        }
        let mut fp = 0u64;
        for l in &clean {
            let events = load_stream(data, &l.file)?;
            let env = AutonomyEnvelope { max_autonomous_risk_tier: 2, max_autonomous_fraction: mf, window: 100 };
            let mut m = AutonomyMeter::new(env);
            let mut any = false;
            for ev in &events {
                if m.observe(ev.case, ev.autonomous, ev.autonomy_level).is_some() { any = true; }
            }
            if any { fp += 1; }
        }
        envelopes.push(json!({
            "max_autonomous_fraction": mf,
            "n_creep": creep.len(), "detected": detected,
            "recall": detected as f64 / creep.len().max(1) as f64,
            "mean_latency": mean(&latencies),
            "latencies": latencies,
            "far": fp as f64 / clean.len().max(1) as f64,
        }));
    }
    Ok(json!({ "envelopes": envelopes, "n_creep": creep.len(), "n_clean": clean.len() }))
}

// ── E4: unsupervised behavioral anomaly ──────────────────────────────────────
fn exp_e4(data: &str, labels: &[Label]) -> anyhow::Result<serde_json::Value> {
    let anomaly: Vec<&Label> = labels.iter().filter(|l| l.kind == "behavior" && l.t0.is_some()).collect();
    let clean: Vec<&Label> = labels.iter().filter(|l| l.kind == "behavior" && l.t0.is_none()).collect();
    let thresholds = [0.2f64, 0.35, 0.5, 0.75, 1.0];
    let mut sweep = Vec::new();
    for &th in &thresholds {
        let mut latencies = Vec::new();
        let mut detected = 0u64;
        for l in &anomaly {
            let events = load_stream(data, &l.file)?;
            let mut b = BehaviorBaseline::new(200, 100, th);
            let mut lat = None;
            for ev in &events {
                if b.observe(ev.case, &ev.tool).is_some() {
                    if let (Some(t0), None) = (l.t0, lat) {
                        if ev.case >= t0 { lat = Some(ev.case - t0); }
                    }
                }
            }
            if let Some(x) = lat { detected += 1; latencies.push(x as f64); }
        }
        let mut fp = 0u64;
        for l in &clean {
            let events = load_stream(data, &l.file)?;
            let mut b = BehaviorBaseline::new(200, 100, th);
            let mut any = false;
            for ev in &events {
                if b.observe(ev.case, &ev.tool).is_some() { any = true; }
            }
            if any { fp += 1; }
        }
        sweep.push(json!({
            "threshold": th,
            "n_anomaly": anomaly.len(), "detected": detected,
            "recall": detected as f64 / anomaly.len().max(1) as f64,
            "mean_latency": mean(&latencies),
            "latencies": latencies,
            "n_clean": clean.len(), "false_pos": fp,
            "fpr": fp as f64 / clean.len().max(1) as f64,
        }));
    }
    Ok(json!({ "sweep": sweep, "n_anomaly": anomaly.len(), "n_clean": clean.len() }))
}

// ── E5: monitoring overhead ──────────────────────────────────────────────────
fn exp_e5(data: &str, labels: &[Label]) -> anyhow::Result<serde_json::Value> {
    // Concatenate all drift-stream scores into one long signal.
    let mut signal: Vec<f64> = Vec::new();
    for l in labels.iter().filter(|l| l.kind == "drift") {
        for ev in load_stream(data, &l.file)? {
            signal.push(ev.score);
        }
    }
    let mut rows = Vec::new();
    for det_name in DRIFT_DETECTORS {
        let mut det = make_detector(det_name, default_thresh(det_name));
        let t0 = std::time::Instant::now();
        for (i, &x) in signal.iter().enumerate() {
            let _ = det.observe(i as u64, x);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let n = signal.len().max(1);
        let mut row = json!({
            "detector": det_name,
            "events": n,
            "total_ms": elapsed * 1e3,
            "us_per_event": elapsed * 1e6 / n as f64,
            "events_per_sec": n as f64 / elapsed.max(1e-9),
        });
        if det_name == "adwin" {
            // Re-run to read the resident bucket count (memory-bound witness).
            let mut a = Adwin::new(default_thresh("adwin"));
            for (i, &x) in signal.iter().enumerate() { let _ = a.observe(i as u64, x); }
            row["adwin_bucket_count"] = json!(a.bucket_count());
        }
        rows.push(row);
    }
    Ok(json!({ "overhead": rows }))
}

fn write_out(out: &str, name: &str, val: &serde_json::Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let path = Path::new(out).join(name);
    std::fs::write(&path, serde_json::to_string_pretty(val)?)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut data = "datasets/011-drift-monitoring".to_string();
    let mut out = "results/011-drift-monitoring".to_string();
    let mut exp = "all".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data" => data = it.next().unwrap(),
            "--out" => out = it.next().unwrap(),
            "--exp" => exp = it.next().unwrap(),
            _ => {}
        }
    }
    let labels = load_labels(&data)?;
    let run = |e: &str| -> anyhow::Result<()> {
        match e {
            "e1" => write_out(&out, "e1_drift.json", &exp_e1(&data, &labels)?),
            "e2" => write_out(&out, "e2_delayed.json", &exp_e2(&data, &labels)?),
            "e3" => write_out(&out, "e3_autonomy.json", &exp_e3(&data, &labels)?),
            "e4" => write_out(&out, "e4_behavior.json", &exp_e4(&data, &labels)?),
            "e5" => write_out(&out, "e5_overhead.json", &exp_e5(&data, &labels)?),
            other => Err(anyhow::anyhow!("unknown experiment '{other}'")),
        }
    };
    if exp == "all" {
        for e in ["e1", "e2", "e3", "e4", "e5"] {
            run(e)?;
        }
    } else {
        run(&exp)?;
    }
    Ok(())
}
