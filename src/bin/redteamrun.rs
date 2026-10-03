//! E1–E5 scorer for paper 015: continuous red-teaming, IR & resilience.
//!
//! Replays the HAARF-RedTeam corpus through the modeled gateway defense
//! (discovery + regression), drives incident runbooks over synthetic streams,
//! and injects faults to check resilience invariants. Off-hot-path, deterministic,
//! no model required.
//!
//! Run:
//!   cargo run --release --bin redteamrun -- e1 --data datasets/015-redteam --out results/015-redteam
//!   cargo run --release --bin redteamrun -- all --data datasets/015-redteam --out results/015-redteam

use qfire::redteam::chaos::{self, GatewayModel};
use qfire::redteam::corpus::{self, coverage};
use qfire::redteam::incident::{self, mean_finite};
use qfire::redteam::regression::{self, Change};
use qfire::redteam::runner::{self, GatewayProbe};
use qfire::redteam::{AttackProbe, FaultKind, IncidentEvent, ResilienceInvariant, TargetLayer};
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

// Incident-response arm latencies (steps per runbook action). Automated runbooks
// fire near-instantly; manual response waits on a human.
const AUTO_LATENCY: u64 = 1;
const MANUAL_LATENCY: u64 = 12;
const DETECT_THRESHOLD: f64 = 0.5;

const MAX_ROUNDS: u32 = 20;
const DRY_ROUNDS: u32 = 3;

fn write_out(out_dir: &str, name: &str, v: &serde_json::Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let path = Path::new(out_dir).join(name);
    std::fs::write(&path, serde_json::to_string_pretty(v)?)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// Wilson 95% CI half-width helper for a rate (used for catch-rate reporting).
fn wilson(p: f64, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.96;
    let n = n as f64;
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let margin = z * ((p * (1.0 - p) / n) + z * z / (4.0 * n * n)).sqrt() / denom;
    ((centre - margin).max(0.0), (centre + margin).min(1.0))
}

// ── E1: continuous vs one-shot discovery ──────────────────────────────────────
fn exp_e1(data: &str) -> anyhow::Result<serde_json::Value> {
    let attacks = corpus::load_corpus(Path::new(data).join("corpus.jsonl"))?;
    let probe = GatewayProbe::standard();
    let one_shot = runner::one_shot(&attacks, &probe);
    let curve = runner::loop_until_dry(&attacks, &probe, MAX_ROUNDS, DRY_ROUNDS);
    let continuous = curve.last().map(|p| p.cumulative_vulns).unwrap_or(0);
    let cov = coverage(&attacks);
    Ok(json!({
        "experiment": "e1_discovery",
        "n_attacks": attacks.len(),
        "one_shot_vulns": one_shot.len(),
        "continuous_vulns": continuous,
        "extra": continuous.saturating_sub(one_shot.len()),
        "rounds_to_dry": curve.len().saturating_sub(1),
        "curve": curve,
        "coverage": cov,
    }))
}

// ── E2: regression catch at the deployment gate ───────────────────────────────
fn exp_e2(data: &str) -> anyhow::Result<serde_json::Value> {
    let attacks = corpus::load_corpus(Path::new(data).join("corpus.jsonl"))?;
    let probe = GatewayProbe::standard();
    // Closed-set = attacks the baseline gateway blocks (built-layer, signature-bearing).
    let closed: Vec<_> = attacks.iter().filter(|a| probe.probe(a).blocked).cloned().collect();

    // Change battery: disable each built layer in turn, plus two multi-layer changes.
    let built: Vec<TargetLayer> = TargetLayer::ALL.iter().copied().filter(|l| l.built()).collect();
    let mut changes: Vec<Change> = built
        .iter()
        .map(|l| Change { id: format!("disable-{l}"), disabled_layers: vec![*l] })
        .collect();
    changes.push(Change {
        id: "disable-firewall+egress".into(),
        disabled_layers: vec![TargetLayer::Firewall, TargetLayer::Egress],
    });
    changes.push(Change {
        id: "disable-policy+identity+oversight".into(),
        disabled_layers: vec![TargetLayer::Policy, TargetLayer::Identity, TargetLayer::Oversight],
    });

    let results: Vec<_> = changes.iter().map(|c| regression::run_gate(&closed, &probe, c)).collect();
    let catch = regression::catch_rate(&results);
    let total_reopened: usize = results.iter().map(|r| r.reopened).sum();
    let (lo, hi) = wilson(catch, total_reopened.max(1));
    Ok(json!({
        "experiment": "e2_regression",
        "closed_set": closed.len(),
        "n_changes": changes.len(),
        "catch_rate": catch,
        "catch_rate_ci": [lo, hi],
        "total_reopened": total_reopened,
        "per_change": results,
    }))
}

// ── E3: incident response (manual vs automated) ───────────────────────────────
#[derive(Deserialize)]
struct IncidentStream {
    is_attack: bool,
    events: Vec<IncidentEvent>,
}

fn load_streams(data: &str) -> anyhow::Result<Vec<IncidentStream>> {
    let text = std::fs::read_to_string(Path::new(data).join("incidents.jsonl"))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line)?);
    }
    Ok(out)
}

fn exp_e3(data: &str) -> anyhow::Result<serde_json::Value> {
    let streams = load_streams(data)?;
    let (mut mttd, mut auto_c, mut auto_r, mut man_c, mut man_r) =
        (vec![], vec![], vec![], vec![], vec![]);
    let mut detected = 0usize;
    let mut attacks = 0usize;
    for s in &streams {
        if s.is_attack {
            attacks += 1;
        }
        let a = incident::respond(&s.events, DETECT_THRESHOLD, AUTO_LATENCY);
        let m = incident::respond(&s.events, DETECT_THRESHOLD, MANUAL_LATENCY);
        if a.detected {
            detected += 1;
            mttd.push(a.mttd);
            auto_c.push(a.mttc);
            auto_r.push(a.mttr);
            man_c.push(m.mttc);
            man_r.push(m.mttr);
        }
    }
    let mean_mttc_auto = mean_finite(&auto_c);
    let mean_mttc_man = mean_finite(&man_c);
    let reduction = if mean_mttc_man > 0.0 {
        100.0 * (mean_mttc_man - mean_mttc_auto) / mean_mttc_man
    } else {
        0.0
    };
    Ok(json!({
        "experiment": "e3_incident",
        "n_streams": streams.len(),
        "attack_streams": attacks,
        "detected": detected,
        "detection_recall": detected as f64 / attacks.max(1) as f64,
        "mean_mttd": mean_finite(&mttd),
        "automated": { "mttc": mean_mttc_auto, "mttr": mean_finite(&auto_r) },
        "manual": { "mttc": mean_mttc_man, "mttr": mean_finite(&man_r) },
        "mttc_reduction_pct": reduction,
        "auto_latency": AUTO_LATENCY,
        "manual_latency": MANUAL_LATENCY,
    }))
}

// ── E4: chaos / resilience ────────────────────────────────────────────────────
#[derive(Deserialize)]
struct FaultScenario {
    id: String,
    arm: String,
    kind: FaultKind,
    invariant: ResilienceInvariant,
    magnitude: f64,
    fail_open: bool,
    max_degradation: f64,
}

fn exp_e4(data: &str) -> anyhow::Result<serde_json::Value> {
    let text = std::fs::read_to_string(Path::new(data).join("faults.jsonl"))?;
    let mut per_fault = Vec::new();
    let (mut prod_held, mut prod_total) = (0usize, 0usize);
    let (mut val_held, mut val_total) = (0usize, 0usize);
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let s: FaultScenario = serde_json::from_str(line)?;
        let model = GatewayModel { fail_open: s.fail_open, max_degradation: s.max_degradation };
        let r = chaos::inject(&model, s.kind, s.invariant, s.magnitude);
        if s.arm == "production" {
            prod_total += 1;
            if r.held {
                prod_held += 1;
            }
        } else {
            val_total += 1;
            // In the validation arm, "caught" means the planted break did NOT hold.
            if !r.held {
                val_held += 1;
            }
        }
        per_fault.push(json!({ "id": s.id, "arm": s.arm, "result": r }));
    }
    Ok(json!({
        "experiment": "e4_resilience",
        "production": { "invariants_held": prod_held, "total": prod_total },
        "validation": { "violations_caught": val_held, "total": val_total },
        "per_fault": per_fault,
    }))
}

// ── E5: cost vs coverage (thoroughness dial) ──────────────────────────────────
fn exp_e5(data: &str) -> anyhow::Result<serde_json::Value> {
    let attacks = corpus::load_corpus(Path::new(data).join("corpus.jsonl"))?;
    let probe = GatewayProbe::standard();
    // Sweep the round budget; record cumulative probes (cost) vs distinct vulns (coverage).
    let full = runner::loop_until_dry(&attacks, &probe, MAX_ROUNDS, MAX_ROUNDS + 1); // no early stop
    let curve: Vec<_> = full
        .iter()
        .map(|p| json!({ "rounds": p.round, "probes": p.probes, "vulns": p.cumulative_vulns }))
        .collect();
    // ATLAS technique coverage + explicit NOT-tested rows for unbuilt layers.
    let cov = coverage(&attacks);
    let not_tested: Vec<String> = TargetLayer::ALL
        .iter()
        .filter(|l| !l.built())
        .map(|l| l.to_string())
        .collect();
    Ok(json!({
        "experiment": "e5_cost",
        "curve": curve,
        "atlas_techniques": cov.atlas_techniques,
        "layers_built": cov.layers_built,
        "layers_total": cov.layers_total,
        "not_tested_layers": not_tested,
        "note": "unbuilt layers (012-014) have no defense; their attacks always bypass and are reported, not silently dropped",
    }))
}

fn main() -> anyhow::Result<()> {
    let mut data = "datasets/015-redteam".to_string();
    let mut out = "results/015-redteam".to_string();
    let mut exp = "all".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data" => data = it.next().unwrap(),
            "--out" => out = it.next().unwrap(),
            e @ ("e1" | "e2" | "e3" | "e4" | "e5" | "all") => exp = e.to_string(),
            "--exp" => exp = it.next().unwrap(),
            other => eprintln!("ignoring arg: {other}"),
        }
    }
    let run = |e: &str| -> anyhow::Result<()> {
        match e {
            "e1" => write_out(&out, "e1_discovery.json", &exp_e1(&data)?),
            "e2" => write_out(&out, "e2_regression.json", &exp_e2(&data)?),
            "e3" => write_out(&out, "e3_incident.json", &exp_e3(&data)?),
            "e4" => write_out(&out, "e4_resilience.json", &exp_e4(&data)?),
            "e5" => write_out(&out, "e5_cost.json", &exp_e5(&data)?),
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
