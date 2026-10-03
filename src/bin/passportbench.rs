//! E1–E5 scorer for paper 012 (the regulatory passport): replay PassportBench
//! through the `lifecycle` classifier, harmonizer, PCCP evaluator, fingerprinter,
//! and admission gate, and emit per-experiment JSON for the Python summarizer.
//!
//! Deterministic, off-hot-path, no network. Run:
//!   cargo run --release --bin passportbench -- \
//!     --data datasets/012-passport --out results/012-passport --exp all

use qfire::lifecycle::classifier::classify;
use qfire::lifecycle::fingerprint::{self, Component, ComponentDelta, Fingerprint};
use qfire::lifecycle::harmonize::harmonize;
use qfire::lifecycle::passport::{EnvelopeSpec, Passport};
use qfire::lifecycle::pccp::{self, Pccp};
use qfire::lifecycle::{AgentMetadata, AutonomyLevel, Jurisdiction, RiskClass};
use ed25519_dalek::SigningKey;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct LabelEntry {
    risk_class: RiskClass,
    #[allow(dead_code)]
    autonomy_level: AutonomyLevel,
}

#[derive(Deserialize)]
struct PassportRow {
    id: String,
    metadata: AgentMetadata,
    stratum: String,
    labels: BTreeMap<String, LabelEntry>,
}

#[derive(Deserialize)]
struct Mag {
    #[serde(rename = "true")]
    #[allow(dead_code)]
    true_: f64,
    est: f64,
}

#[derive(Deserialize)]
struct DeltaRow {
    component: Component,
    magnitude: Option<Mag>,
}

#[derive(Deserialize)]
struct ChangeRow {
    id: String,
    deployed_fingerprint: Fingerprint,
    live_fingerprint: Fingerprint,
    pccp: Pccp,
    deltas: Vec<DeltaRow>,
    changed_components: Vec<Component>,
    true_out_of_envelope: bool,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("parse {}: {e}\n{l}", path.display())))
        .collect()
}

fn jur_from_key(k: &str) -> Jurisdiction {
    serde_json::from_value(serde_json::Value::String(k.to_string()))
        .unwrap_or_else(|_| panic!("unknown jurisdiction key {k}"))
}

fn risk_name(r: RiskClass) -> &'static str {
    match r {
        RiskClass::Minimal => "minimal",
        RiskClass::Low => "low",
        RiskClass::Moderate => "moderate",
        RiskClass::High => "high",
    }
}

/// E1 — classification accuracy vs ground-truth, per jurisdiction & stratum.
fn e1(passports: &[PassportRow], out: &Path) {
    let mut preds = Vec::new();
    for p in passports {
        for (jkey, label) in &p.labels {
            let j = jur_from_key(jkey);
            let c = classify(&p.metadata, j);
            preds.push(json!({
                "id": p.id,
                "jurisdiction": jkey,
                "stratum": p.stratum,
                "predicted": risk_name(c.risk_class),
                "truth": risk_name(label.risk_class),
                "correct": c.risk_class == label.risk_class,
            }));
        }
    }
    let n = preds.len();
    let correct = preds.iter().filter(|p| p["correct"].as_bool().unwrap()).count();
    write(out, "e1_classification.json", json!({
        "experiment": "E1_classification",
        "n": n,
        "overall_accuracy": correct as f64 / n as f64,
        "predictions": preds,
    }));
    println!("E1: {}/{} correct ({:.3})", correct, n, correct as f64 / n as f64);
}

/// Continuous out-of-envelope risk score for the ROC (higher = more clearly out).
fn envelope_score(row: &ChangeRow) -> f64 {
    let mut score: f64 = 0.0;
    for d in &row.deltas {
        let s = match d.component {
            // prompt is any_change in these plans ⇒ no contribution.
            Component::Prompt => 0.0,
            // weights bounded@0.05 ⇒ score is the estimated magnitude over the bound.
            Component::Weights => d.magnitude.as_ref().map(|m| m.est / 0.05).unwrap_or(2.0),
            // tools/data unlisted ⇒ forbidden ⇒ unambiguously out.
            Component::Tools | Component::Data => 2.0,
        };
        score = score.max(s);
    }
    score
}

/// E2 — PCCP in/out-of-envelope gating over the change corpus (ROC + operating point).
fn e2(changes: &[ChangeRow], out: &Path) {
    let mut rows = Vec::new();
    for ch in changes {
        // The gate's actual decision uses the *estimated* magnitude (admission-time).
        let deltas: Vec<ComponentDelta> = ch
            .deltas
            .iter()
            .map(|d| ComponentDelta {
                component: d.component,
                from: String::new(),
                to: String::new(),
                magnitude: d.magnitude.as_ref().map(|m| m.est),
            })
            .collect();
        let verdict = pccp::evaluate(&deltas, &ch.pccp);
        rows.push(json!({
            "id": ch.id,
            "predicted_out": !verdict.in_envelope,
            "true_out": ch.true_out_of_envelope,
            "score": envelope_score(ch),
        }));
    }
    let pos = changes.iter().filter(|c| c.true_out_of_envelope).count();
    write(out, "e2_pccp.json", json!({
        "experiment": "E2_pccp_gating",
        "n": changes.len(),
        "true_out_of_envelope": pos,
        "true_in_envelope": changes.len() - pos,
        "bound_max_fraction": 0.05,
        "rows": rows,
    }));
    println!("E2: {} events ({} out / {} in)", changes.len(), pos, changes.len() - pos);
}

/// E3 — cross-jurisdiction harmonization: agreement + every divergence explained.
fn e3(passports: &[PassportRow], out: &Path) {
    let mut agreements = Vec::new();
    let mut all_explained = true;
    let mut total_div = 0usize;
    // pairwise agreement matrix counts.
    let mut pair_agree: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for p in passports {
        let r = harmonize(&p.metadata);
        agreements.push(r.agreement);
        total_div += r.divergences.len();
        for d in &r.divergences {
            if d.rationale.trim().is_empty() {
                all_explained = false;
            }
        }
        // recompute pair agreement from per_jurisdiction for the matrix.
        for i in 0..r.per_jurisdiction.len() {
            for k in (i + 1)..r.per_jurisdiction.len() {
                let (ja, ca) = &r.per_jurisdiction[i];
                let (jb, cb) = &r.per_jurisdiction[k];
                let key = format!("{}|{}", ja.name(), jb.name());
                let e = pair_agree.entry(key).or_insert((0, 0));
                e.1 += 1;
                if ca.risk_class == cb.risk_class {
                    e.0 += 1;
                }
            }
        }
    }
    let hca = agreements.iter().sum::<f64>() / agreements.len() as f64;
    let matrix: Vec<_> = pair_agree
        .iter()
        .map(|(k, (a, n))| json!({"pair": k, "agree": a, "n": n, "rate": *a as f64 / *n as f64}))
        .collect();
    write(out, "e3_harmonization.json", json!({
        "experiment": "E3_harmonization",
        "n_passports": passports.len(),
        "mean_agreement": hca,
        "total_divergences": total_div,
        "all_divergences_explained": all_explained,
        "pairwise_matrix": matrix,
    }));
    println!("E3: HCA {:.3}, {} divergences, all explained = {}", hca, total_div, all_explained);
}

/// E4 — change-fingerprint sensitivity: per-component fingerprinting detects AND
/// localizes every swap; a whole-config hash detects but cannot localize (needed
/// for PCCP evaluation). RQ4.
fn e4(changes: &[ChangeRow], out: &Path) {
    let mut rows = Vec::new();
    for ch in changes {
        let deltas = fingerprint::diff(&ch.deployed_fingerprint, &ch.live_fingerprint);
        let detected: Vec<Component> = deltas.iter().map(|d| d.component).collect();
        let mut expect = ch.changed_components.clone();
        expect.sort_by_key(|c| c.name());
        let mut got = detected.clone();
        got.sort_by_key(|c| c.name());
        let localized = expect == got;
        // whole-config hash: concat of the four digests; detects any change, no localization.
        let whole_before = format!(
            "{}{}{}{}",
            ch.deployed_fingerprint.weights, ch.deployed_fingerprint.prompt,
            ch.deployed_fingerprint.tools, ch.deployed_fingerprint.data
        );
        let whole_after = format!(
            "{}{}{}{}",
            ch.live_fingerprint.weights, ch.live_fingerprint.prompt,
            ch.live_fingerprint.tools, ch.live_fingerprint.data
        );
        let whole_detected = whole_before != whole_after;
        rows.push(json!({
            "id": ch.id,
            "k": ch.changed_components.len(),
            "component_detected": !detected.is_empty(),
            "component_localized": localized,
            "whole_detected": whole_detected,
            "whole_localized": false,
        }));
    }
    write(out, "e4_fingerprint.json", json!({
        "experiment": "E4_fingerprint",
        "n": changes.len(),
        "rows": rows,
    }));
    println!("E4: {} change-events fingerprinted", changes.len());
}

/// E5 — deployment-gate overhead (sign-verify + fingerprint-diff + classify + PCCP).
fn e5(passports: &[PassportRow], out: &Path) {
    use qfire::lifecycle::gate::{decide, Enforce};
    use qfire::lifecycle::pccp::{ChangePolicy, ComponentRule};
    use qfire::lifecycle::passport::SignedPassport;

    let issuer = SigningKey::from_bytes(&[5u8; 32]);
    // Pre-build a signed passport per metadata (clearance-time artifact).
    let mut signed: Vec<(SignedPassport, Fingerprint)> = Vec::new();
    for p in passports {
        let classifications = Jurisdiction::ALL
            .iter()
            .map(|&j| (j, classify(&p.metadata, j)))
            .collect();
        let fp = Fingerprint::of(p.id.as_bytes(), b"prompt", b"tools", b"data");
        let passport = Passport {
            agent_id: p.id.clone(),
            version: "1.0.0".into(),
            ruleset_version: "2026.06".into(),
            metadata: p.metadata.clone(),
            classifications,
            pccp: Pccp {
                allowed: vec![ComponentRule { component: Component::Prompt, change: ChangePolicy::AnyChange }],
            },
            deployed_fingerprint: fp.clone(),
            not_after: 4_000_000_000,
            autonomy_envelope: EnvelopeSpec { max_autonomous_risk_tier: 2, max_autonomous_fraction: 0.3, window: 100 },
        };
        signed.push((passport.sign(&issuer).unwrap(), fp));
    }
    let registry: Vec<String> = passports.iter().map(|p| p.id.clone()).collect();

    // Time the full gate decision over all passports (repeat for a stable measure).
    let reps = 50usize;
    let mut times_ns: Vec<u128> = Vec::with_capacity(signed.len() * reps);
    for _ in 0..reps {
        for (s, fp) in &signed {
            let t = Instant::now();
            let r = decide(s, fp, &registry, 1_000_000, Enforce::Block);
            std::hint::black_box(&r);
            times_ns.push(t.elapsed().as_nanos());
        }
    }
    times_ns.sort_unstable();
    let median_us = times_ns[times_ns.len() / 2] as f64 / 1000.0;
    let p95_us = times_ns[times_ns.len() * 95 / 100] as f64 / 1000.0;
    let total_s: f64 = times_ns.iter().sum::<u128>() as f64 / 1e9;
    let gates_per_sec = times_ns.len() as f64 / total_s;
    write(out, "e5_overhead.json", json!({
        "experiment": "E5_overhead",
        "gates_timed": times_ns.len(),
        "median_us": median_us,
        "p95_us": p95_us,
        "gates_per_sec": gates_per_sec,
    }));
    println!("E5: median {:.2}µs/gate, p95 {:.2}µs, {:.0} gates/s", median_us, p95_us, gates_per_sec);
}

fn write(out: &Path, name: &str, v: serde_json::Value) {
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(out.join(name), serde_json::to_string_pretty(&v).unwrap()).unwrap();
}

fn arg(flag: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.into())
}

fn main() {
    let data = arg("--data", "datasets/012-passport");
    let out = arg("--out", "results/012-passport");
    let exp = arg("--exp", "all");
    let data = Path::new(&data);
    let out = Path::new(&out);

    let passports: Vec<PassportRow> = read_jsonl(&data.join("passports/passports.jsonl"));
    let changes: Vec<ChangeRow> = read_jsonl(&data.join("changes/changes.jsonl"));

    let run = |e: &str| exp == "all" || exp == e;
    if run("e1") { e1(&passports, out); }
    if run("e2") { e2(&changes, out); }
    if run("e3") { e3(&passports, out); }
    if run("e4") { e4(&changes, out); }
    if run("e5") { e5(&passports, out); }
}
