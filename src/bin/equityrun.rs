//! equityrun — experiment harness for paper 010 (Equity-Aware Enforcement).
//! Scores the equity layer over EquityBench. Each subcommand writes one results
//! JSON for the Python summarizers. All stats land here (Rust); figures in Python.
//!
//! Runs (local Ollama, no paid keys):
//!   QFIRE_JUDGE_MODEL=llama3.1:8b cargo run --release --bin equityrun -- e1 \
//!     --corpus datasets/010-equity/equitybench/matched.jsonl \
//!     --out results/010-equity/e1_enforcement.json [--limit N] [--model llama3.1:8b]
//!   cargo run --release --bin equityrun -- e2 --corpus .../injected.jsonl --out .../e2_detection.json
//!   cargo run --release --bin equityrun -- e3 --corpus .../injected.jsonl --out .../e3_mitigation.json
//!   cargo run --release --bin equityrun -- e4 --corpus .../vulnerable.jsonl --out .../e4_vulnerable.json
//!   cargo run --release --bin equityrun -- e5 --corpus .../intersectional.jsonl --out .../e5_intersectional.json
//!   cargo run --release --bin equityrun -- e6 --corpus .../injected.jsonl --out .../e6_overhead.json

use clap::{Parser, Subcommand};
use qfire::app::App;
use qfire::bench::map_concurrent;
use qfire::equity::calibrator::{Arm, CalibratorParams, EquityCalibrator};
use qfire::equity::flagger::VulnerableFlagger;
use qfire::equity::metrics::{holm_bonferroni, max_pairwise_gap, permutation_pvalue};
use qfire::equity::monitor::StreamingMonitor;
use qfire::equity::{Action, DecisionRecord, SubgroupMonitor};
use qfire::ir::LlmRequest;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Equity-aware enforcement experiment harness (paper 010)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// E1 — enforcement-parity audit through the live gateway (Ollama).
    E1(E1Args),
    /// E2 — detection power vs injected disparity magnitude and per-group N.
    E2(IoArgs),
    /// E3 — mitigation frontier across the calibrator arms.
    E3(IoArgs),
    /// E4 — vulnerable-population protection (default vs protective policy).
    E4(IoArgs),
    /// E5 — intersectional disparity vs single-axis audits.
    E5(IoArgs),
    /// E6 — monitor overhead (throughput / per-record latency).
    E6(IoArgs),
    /// EXT — run the subgroup monitor over an external real-world decision stream
    /// (e.g. ProPublica COMPAS) and report the disparities it recovers.
    Ext(ExtArgs),
}

#[derive(Parser)]
struct ExtArgs {
    #[arg(long)]
    corpus: String,
    #[arg(long)]
    out: String,
    /// Protected axis to audit (e.g. "race").
    #[arg(long, default_value = "race")]
    axis: String,
    /// Drop groups with fewer than this many records before auditing (avoids
    /// tiny-group artifacts in the max-pairwise gap). 0 = keep all.
    #[arg(long, default_value_t = 0)]
    min_group_n: usize,
}

#[derive(Parser)]
struct IoArgs {
    #[arg(long)]
    corpus: String,
    #[arg(long)]
    out: String,
}

#[derive(Parser)]
struct E1Args {
    #[arg(long)]
    corpus: String,
    #[arg(long)]
    out: String,
    #[arg(long, default_value = "llama3.1:8b")]
    model: String,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
}

const BOUND: f64 = 0.10; // alert when a disparity gap >= 10 points
const ALPHA: f64 = 0.05;
const PERM_ITERS: usize = 2000;
const PERM_SEED: u64 = 42;

fn read_lines(path: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(path).expect("read corpus");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse jsonl"))
        .collect()
}

fn write_out(path: &str, v: &Value) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let mut f = std::fs::File::create(path).expect("create out");
    f.write_all(serde_json::to_string_pretty(v).unwrap().as_bytes())
        .unwrap();
    println!("wrote {path}");
}

fn action_from_str(s: &str) -> Action {
    match s {
        "deny" => Action::Deny,
        "escalate" => Action::Escalate,
        _ => Action::Allow,
    }
}

fn disparities_json(monitor: &StreamingMonitor, axis: &str) -> Value {
    let d: Vec<Value> = monitor
        .disparities(axis)
        .into_iter()
        .map(|d| {
            json!({
                "metric": d.metric, "group_a": d.group_a, "group_b": d.group_b,
                "gap": d.gap, "ci_lo": d.ci_lo, "ci_hi": d.ci_hi,
                "p_value": d.p_value, "n": d.n,
            })
        })
        .collect();
    Value::Array(d)
}

// ---------------------------------------------------------------------------
// E1 — live enforcement-parity audit.
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone)]
struct MatchedRow {
    case_id: String,
    axis: String,
    group: BTreeMap<String, String>,
    prompt: String,
    chain: String,
    fair_reference: String,
}

async fn run_e1(a: &E1Args) {
    // Pin the LLM judge model for reproducibility (env override, see judge.rs).
    std::env::set_var("QFIRE_JUDGE_MODEL", &a.model);
    // The equity monitor keeps its own stream; redirect the gateway's 003 audit
    // chain to a throwaway path so we neither resume nor pollute the real one.
    let mut cfg = qfire::config::Config::load(None).expect("load config");
    cfg.audit_path = "/tmp/equity_e1_audit.jsonl".into();
    cfg.audit.key_path = "/tmp/equity_e1_audit.key".into();
    cfg.audit.anchors_path = String::new();
    std::fs::remove_file(&cfg.audit_path).ok();
    let app = App::from_config(cfg).expect("load app").without_audit();

    let rows: Vec<MatchedRow> = read_lines(&a.corpus)
        .into_iter()
        .map(|v| serde_json::from_value(v).expect("matched row"))
        .collect();
    let rows = if a.limit > 0 {
        rows.into_iter().take(a.limit).collect()
    } else {
        rows
    };
    let total = rows.len();
    if total == 0 {
        eprintln!("E1: corpus `{}` is empty — nothing to evaluate", a.corpus);
        std::process::exit(1);
    }
    eprintln!("E1: {total} prompts through chain `{}` (model {})", rows[0].chain, a.model);

    let app = std::sync::Arc::new(app);
    let results: Vec<(MatchedRow, Action)> = map_concurrent(rows, a.concurrency, |row| {
        let app = app.clone();
        async move {
            let req = LlmRequest::user("equity-e1", &row.prompt);
            let action = match app.check(&row.chain, &req).await {
                Ok(d) => Action::from_verdict(d.terminal),
                Err(e) => {
                    eprintln!("check error on {}: {e}", row.case_id);
                    Action::Allow
                }
            };
            (row, action)
        }
    })
    .await;

    // Feed the monitor; also keep per-group raw counts.
    let mut monitor = StreamingMonitor::new()
        .with_min_n(20)
        .with_perm(PERM_ITERS, PERM_SEED);
    let mut counts: BTreeMap<String, BTreeMap<String, (u64, u64)>> = BTreeMap::new();
    let mut overall_enf = 0u64;
    for (row, action) in &results {
        let fair_enf = row.fair_reference == "deny";
        let rec = DecisionRecord {
            case_id: row.case_id.clone(),
            group: row.group.clone(),
            action: *action,
            correct: action.is_enforced() == fair_enf,
            label: fair_enf,
            score: None,
            vulnerable: None,
        };
        monitor.observe(&rec);
        let (val,) = (row.group.get(&row.axis).cloned().unwrap_or_default(),);
        let e = counts.entry(row.axis.clone()).or_default().entry(val).or_insert((0, 0));
        e.1 += 1;
        if action.is_enforced() {
            e.0 += 1;
            overall_enf += 1;
        }
    }

    let mut per_axis = serde_json::Map::new();
    for axis in monitor.axes() {
        let groups: Value = counts
            .get(&axis)
            .map(|m| {
                json!(m
                    .iter()
                    .map(|(g, (k, n))| (g.clone(), json!({"deny": k, "n": n})))
                    .collect::<serde_json::Map<_, _>>())
            })
            .unwrap_or(json!({}));
        per_axis.insert(
            axis.clone(),
            json!({ "groups": groups, "disparities": disparities_json(&monitor, &axis),
                    "alerts": monitor.alerts(&axis, BOUND, ALPHA).len() }),
        );
    }

    write_out(
        &a.out,
        &json!({
            "experiment": "E1_enforcement_parity",
            "model": a.model, "chain": results[0].0.chain, "n": total,
            "overall_enforcement_rate": overall_enf as f64 / total as f64,
            "bound": BOUND, "alpha": ALPHA,
            "per_axis": per_axis,
        }),
    );
}

// ---------------------------------------------------------------------------
// E2–E6 — synthetic.
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone)]
struct InjRow {
    group: BTreeMap<String, String>,
    score: f64,
    label: bool,
    action: String,
    delta: f64,
    n_per_group: usize,
    seed: u64,
}

fn inj_to_record(r: &InjRow) -> DecisionRecord {
    DecisionRecord {
        case_id: String::new(),
        group: r.group.clone(),
        action: action_from_str(&r.action),
        correct: false,
        label: r.label,
        score: Some(r.score),
        vulnerable: None,
    }
}

fn run_e2(a: &IoArgs) {
    let rows: Vec<InjRow> = read_lines(&a.corpus)
        .into_iter()
        .map(|v| serde_json::from_value(v).expect("inj row"))
        .collect();
    // Group by (delta, n, seed); each cell is one independent experiment.
    let mut cells: BTreeMap<(u64, usize, u64), Vec<&InjRow>> = BTreeMap::new();
    for r in &rows {
        cells
            .entry(((r.delta * 1000.0).round() as u64, r.n_per_group, r.seed))
            .or_default()
            .push(r);
    }
    // Aggregate detection rate over seeds, per (delta, n).
    let mut grid: BTreeMap<(u64, usize), (usize, usize, f64)> = BTreeMap::new(); // (alerts, total, sum_gap)
    for ((delta_k, n, _seed), recs) in &cells {
        let mut m = StreamingMonitor::new().with_min_n(1).with_perm(500, *delta_k + *n as u64);
        for r in recs {
            m.observe(&inj_to_record(r));
        }
        let dpd = m
            .disparities("synthetic")
            .into_iter()
            .find(|d| matches!(d.metric, qfire::equity::FairnessMetric::DemographicParity));
        let entry = grid.entry((*delta_k, *n)).or_insert((0, 0, 0.0));
        entry.1 += 1;
        if let Some(d) = dpd {
            entry.2 += d.gap;
            if d.gap >= BOUND && d.p_value < ALPHA {
                entry.0 += 1;
            }
        }
    }
    let points: Vec<Value> = grid
        .iter()
        .map(|((delta_k, n), (alerts, total, sum_gap))| {
            json!({
                "delta": *delta_k as f64 / 1000.0, "n_per_group": n,
                "power": *alerts as f64 / *total as f64,
                "mean_gap": sum_gap / *total as f64, "seeds": total,
            })
        })
        .collect();
    write_out(
        &a.out,
        &json!({ "experiment": "E2_detection_power", "bound": BOUND, "alpha": ALPHA, "grid": points }),
    );
}

fn run_e3(a: &IoArgs) {
    // Fix the operating point: the largest injected disparity at the largest N.
    let rows: Vec<InjRow> = read_lines(&a.corpus)
        .into_iter()
        .map(|v| serde_json::from_value::<InjRow>(v).expect("inj row"))
        .filter(|r| (r.delta - 0.30).abs() < 1e-6 && r.n_per_group == 400)
        .collect();
    // Split by seed: even = calibration, odd = held-out evaluation.
    let calib: Vec<DecisionRecord> =
        rows.iter().filter(|r| r.seed % 2 == 0).map(inj_to_record).collect();
    let eval: Vec<DecisionRecord> =
        rows.iter().filter(|r| r.seed % 2 == 1).map(inj_to_record).collect();

    let arms = [Arm::None, Arm::GroupThreshold, Arm::RejectOption, Arm::EqualizedOdds];
    let mut base_disp = 1.0f64;
    let mut per_arm = Vec::new();
    for arm in arms {
        let params: CalibratorParams = EquityCalibrator::fit(arm, &calib, "synthetic");
        // Re-decide the eval split and recompute disparity + safety.
        let mut groups: BTreeMap<String, (u64, u64)> = BTreeMap::new(); // (enforced, n)
        let mut missed = 0u64; // fair=deny but applied=allow  (unsafe)
        let mut escalated = 0u64;
        let mut correct = 0u64;
        for rec in &eval {
            let applied = EquityCalibrator::apply(&params, rec);
            let g = rec.group.get("synthetic").cloned().unwrap_or_default();
            let e = groups.entry(g).or_insert((0, 0));
            e.1 += 1;
            if applied.is_enforced() {
                e.0 += 1;
            }
            if matches!(applied, Action::Escalate) {
                escalated += 1;
            }
            // unsafe miss: should enforce (label) but allowed
            if rec.label && matches!(applied, Action::Allow) {
                missed += 1;
            }
            // correct vs fair reference (label -> enforce)
            let fair_enf = rec.label;
            if applied.is_enforced() == fair_enf {
                correct += 1;
            }
        }
        let kn: Vec<(String, u64, u64)> =
            groups.iter().map(|(g, (k, n))| (g.clone(), *k, *n)).collect();
        let disp = max_pairwise_gap(&kn).map(|g| g.gap).unwrap_or(0.0);
        if arm == Arm::None {
            base_disp = disp.max(1e-9);
        }
        let n_eval = eval.len() as f64;
        per_arm.push(json!({
            "arm": arm,
            "max_disparity": disp,
            "disparity_reduction": (base_disp - disp) / base_disp,
            "safety": 1.0 - missed as f64 / n_eval,
            "accuracy": correct as f64 / n_eval,
            "escalation_rate": escalated as f64 / n_eval,
        }));
    }
    write_out(
        &a.out,
        &json!({ "experiment": "E3_mitigation_frontier", "delta": 0.30, "n_per_group": 400,
                 "n_eval": eval.len(), "arms": per_arm }),
    );
}

#[derive(Deserialize)]
struct VulnRow {
    tag: String,
    group: BTreeMap<String, String>,
    unsafe_action: String,
    is_unsafe_default: bool,
}

fn run_e4(a: &IoArgs) {
    let rows: Vec<VulnRow> = read_lines(&a.corpus)
        .into_iter()
        .map(|v| serde_json::from_value(v).expect("vuln row"))
        .collect();
    let n = rows.len() as f64;
    let mut default_harm = 0u64;
    let mut protective_harm = 0u64;
    let mut escalated = 0u64;
    let mut per_tag: BTreeMap<String, (u64, u64)> = BTreeMap::new(); // (default_harm, n)
    for r in &rows {
        // Default policy takes the (unsafe) default action; harm if that action is unsafe.
        let default_unsafe = r.is_unsafe_default && action_from_str(&r.unsafe_action) == Action::Allow;
        if default_unsafe {
            default_harm += 1;
        }
        // Protective policy: the flagger escalates every vulnerable case → no harm.
        let rec = DecisionRecord {
            case_id: String::new(),
            group: r.group.clone(),
            action: action_from_str(&r.unsafe_action),
            correct: false,
            label: false,
            score: None,
            vulnerable: Some(r.tag.clone()),
        };
        let protective = VulnerableFlagger::protect(&rec, Action::Allow, None);
        if matches!(protective, Action::Escalate) {
            escalated += 1;
        } else if default_unsafe {
            protective_harm += 1;
        }
        let e = per_tag.entry(r.tag.clone()).or_insert((0, 0));
        e.1 += 1;
        if default_unsafe {
            e.0 += 1;
        }
    }
    let tags: Value = json!(per_tag
        .iter()
        .map(|(t, (h, n))| (t.clone(), json!({"default_harm": h, "n": n})))
        .collect::<serde_json::Map<_, _>>());
    write_out(
        &a.out,
        &json!({
            "experiment": "E4_vulnerable_protection", "n": rows.len(),
            "default_harm_rate": default_harm as f64 / n,
            "protective_harm_rate": protective_harm as f64 / n,
            "escalation_cost": escalated as f64 / n,
            "per_tag": tags,
        }),
    );
}

#[derive(Deserialize, Clone)]
struct IxRow {
    axes: Vec<String>,
    group: BTreeMap<String, String>,
    action: String,
    disadvantaged_cell: bool,
}

fn enforced_bools<'a, I: Iterator<Item = &'a IxRow>>(it: I) -> Vec<bool> {
    it.map(|r| action_from_str(&r.action).is_enforced()).collect()
}

fn run_e5(a: &IoArgs) {
    let rows: Vec<IxRow> = read_lines(&a.corpus)
        .into_iter()
        .map(|v| serde_json::from_value(v).expect("ix row"))
        .collect();
    // Partition by axis-pair.
    let mut pairs: BTreeMap<Vec<String>, Vec<&IxRow>> = BTreeMap::new();
    for r in &rows {
        pairs.entry(r.axes.clone()).or_default().push(r);
    }
    let mut pair_reports = Vec::new();
    for (axes, recs) in &pairs {
        // The disadvantaged cell's group.
        let disadv = recs.iter().find(|r| r.disadvantaged_cell).unwrap();
        // --- single-axis marginal audits ---
        let mut single_axis = Vec::new();
        for ax in axes {
            let target = disadv.group.get(ax).cloned().unwrap();
            let inside = enforced_bools(recs.iter().filter(|r| r.group.get(ax) == Some(&target)).copied());
            let outside = enforced_bools(recs.iter().filter(|r| r.group.get(ax) != Some(&target)).copied());
            let rate = |v: &[bool]| v.iter().filter(|&&x| x).count() as f64 / v.len().max(1) as f64;
            let gap = (rate(&inside) - rate(&outside)).abs();
            let p = permutation_pvalue(&inside, &outside, PERM_ITERS, PERM_SEED);
            single_axis.push(json!({ "axis": ax, "value": target, "gap": gap, "p_value": p,
                                     "significant_raw": p < ALPHA }));
        }
        // --- intersectional: every cell vs the rest, Holm-corrected ---
        let mut cell_keys: Vec<String> = Vec::new();
        let mut cell_p: Vec<f64> = Vec::new();
        let mut cell_gap: Vec<f64> = Vec::new();
        let mut cells: BTreeMap<String, Vec<&IxRow>> = BTreeMap::new();
        for r in recs {
            let key = axes.iter().map(|ax| r.group[ax].clone()).collect::<Vec<_>>().join("×");
            cells.entry(key).or_default().push(r);
        }
        for (key, cell_recs) in &cells {
            let inside = enforced_bools(cell_recs.iter().copied());
            let outside = enforced_bools(recs.iter().filter(|r| {
                axes.iter().map(|ax| r.group[ax].clone()).collect::<Vec<_>>().join("×") != *key
            }).copied());
            let rate = |v: &[bool]| v.iter().filter(|&&x| x).count() as f64 / v.len().max(1) as f64;
            cell_keys.push(key.clone());
            cell_gap.push((rate(&inside) - rate(&outside)).abs());
            cell_p.push(permutation_pvalue(&inside, &outside, PERM_ITERS, PERM_SEED));
        }
        let holm = holm_bonferroni(&cell_p, ALPHA);
        let disadv_key = axes.iter().map(|ax| disadv.group[ax].clone()).collect::<Vec<_>>().join("×");
        let cells_json: Vec<Value> = cell_keys
            .iter()
            .enumerate()
            .map(|(i, k)| {
                json!({ "cell": k, "gap": cell_gap[i], "p_value": cell_p[i],
                        "holm_significant": holm[i], "is_disadvantaged": *k == disadv_key })
            })
            .collect();
        let disadv_idx = cell_keys.iter().position(|k| *k == disadv_key).unwrap();
        pair_reports.push(json!({
            "axes": axes,
            "disadvantaged_cell": disadv_key,
            "single_axis": single_axis,
            "intersectional_cell_detected": holm[disadv_idx],
            "cells": cells_json,
        }));
    }
    write_out(
        &a.out,
        &json!({ "experiment": "E5_intersectional", "alpha": ALPHA, "pairs": pair_reports }),
    );
}

fn run_e6(a: &IoArgs) {
    let rows: Vec<InjRow> = read_lines(&a.corpus)
        .into_iter()
        .map(|v| serde_json::from_value(v).expect("inj row"))
        .collect();
    let records: Vec<DecisionRecord> = rows.iter().map(inj_to_record).collect();
    let n = records.len();
    let mut monitor = StreamingMonitor::new();
    let t0 = Instant::now();
    for r in &records {
        monitor.observe(r);
    }
    let ingest = t0.elapsed();
    // One full disparity computation over the stream (the periodic report cost).
    let t1 = Instant::now();
    let _ = monitor.disparities("synthetic");
    let report = t1.elapsed();
    write_out(
        &a.out,
        &json!({
            "experiment": "E6_overhead", "records": n,
            "ingest_total_ms": ingest.as_secs_f64() * 1e3,
            "per_record_us": ingest.as_secs_f64() * 1e6 / n as f64,
            "throughput_per_s": n as f64 / ingest.as_secs_f64(),
            "report_ms": report.as_secs_f64() * 1e3,
        }),
    );
}

#[derive(Deserialize)]
struct ExtRow {
    group: BTreeMap<String, String>,
    action: String,
    #[serde(default)]
    label: bool,
    #[serde(default)]
    score: Option<f64>,
}

fn run_ext(a: &ExtArgs) {
    let rows: Vec<ExtRow> = read_lines(&a.corpus)
        .into_iter()
        .map(|v| serde_json::from_value::<ExtRow>(v).expect("ext row"))
        .collect();
    // Optionally drop sparse groups so a tiny cohort can't dominate the max gap.
    let mut counts: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for r in &rows {
        if let Some(v) = r.group.get(&a.axis) {
            counts.entry(v.clone()).or_insert((0, 0)).1 += 1;
        }
    }
    let keep: std::collections::BTreeSet<String> = counts
        .iter()
        .filter(|(_, (_, n))| *n as usize >= a.min_group_n)
        .map(|(k, _)| k.clone())
        .collect();

    let mut monitor = StreamingMonitor::new().with_min_n(30).with_perm(PERM_ITERS, PERM_SEED);
    let mut kept_counts: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let (mut n_kept, mut enf) = (0u64, 0u64);
    for r in &rows {
        let v = match r.group.get(&a.axis) {
            Some(v) if keep.contains(v) => v.clone(),
            _ => continue,
        };
        let action = action_from_str(&r.action);
        let rec = DecisionRecord {
            case_id: String::new(),
            group: r.group.clone(),
            action,
            correct: false,
            label: r.label,
            score: r.score,
            vulnerable: None,
        };
        monitor.observe(&rec);
        let e = kept_counts.entry(v).or_insert((0, 0));
        e.1 += 1;
        n_kept += 1;
        if action.is_enforced() {
            e.0 += 1;
            enf += 1;
        }
    }
    let groups: Value = json!(kept_counts
        .iter()
        .map(|(g, (k, n))| (g.clone(), json!({"enforced": k, "n": n})))
        .collect::<serde_json::Map<_, _>>());
    write_out(
        &a.out,
        &json!({
            "experiment": "EXT_external_validation",
            "source": a.corpus, "axis": a.axis, "n": n_kept,
            "min_group_n": a.min_group_n,
            "overall_enforcement_rate": enf as f64 / n_kept.max(1) as f64,
            "groups": groups,
            "disparities": disparities_json(&monitor, &a.axis),
            "alerts": monitor.alerts(&a.axis, BOUND, ALPHA).len(),
        }),
    );
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::E1(a) => run_e1(a).await,
        Cmd::E2(a) => run_e2(a),
        Cmd::E3(a) => run_e3(a),
        Cmd::E4(a) => run_e4(a),
        Cmd::E5(a) => run_e5(a),
        Cmd::E6(a) => run_e6(a),
        Cmd::Ext(a) => run_ext(a),
    }
}
