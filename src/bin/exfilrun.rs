//! E1/E2 scorer for paper 006: load ExfilBench, run each detector over every session's outbound
//! call, and record per-(detector, attack_type) prevention (did the detector flag a leaking call)
//! and false-blocks on legit sessions. Output JSON for the Python summarizer.
//!
//! Run: cargo run --release --bin exfilrun -- \
//!   --corpus datasets/006-phi-egress/exfilbench/sessions.jsonl \
//!   --ner-model llama3.1:8b --out results/006-phi-egress/e1e2.json

use qfire::config::Config;
use qfire::egress::content::{LlmNerDetector, RegexDlp};
use qfire::egress::kanon::QUASI_IDENTIFIERS;
use qfire::egress::taint::{TaintDetector, TaintStore};
use qfire::egress::Detector;
use qfire::provider::ProviderRegistry;
use serde_json::Value;
use std::collections::BTreeMap;

fn arg_strings(args: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Some(map) = args.as_object() {
        for (k, v) in map {
            out.push((k.clone(), v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string())));
        }
    }
    out
}

/// Extract the call sequence for a session as a list of (arg, value) pair lists. Uses the row's
/// `calls` array when present, otherwise falls back to the single `outbound.args` object.
fn session_calls(s: &Value) -> Vec<Vec<(String, String)>> {
    if let Some(arr) = s["calls"].as_array() {
        arr.iter().map(arg_strings).collect()
    } else {
        vec![arg_strings(&s["outbound"]["args"])]
    }
}

/// Mirrors `egress::anomaly::SessionMonitor`: sum, over all calls whose `"to"` value is EXTERNAL
/// (non-empty and not containing "hospital.org"), of the byte lengths of all non-`"to"` values.
fn external_bytes(calls: &[Vec<(String, String)>]) -> usize {
    let mut total = 0usize;
    for call in calls {
        let to = call.iter().find(|(k, _)| k == "to").map(|(_, v)| v.as_str()).unwrap_or("");
        let external = !to.is_empty() && !to.contains("hospital.org");
        if !external {
            continue;
        }
        for (k, v) in call {
            if k != "to" {
                total += v.len();
            }
        }
    }
    total
}

/// Count of quasi-identifier field names present in a single call's args.
fn qi_count(call: &[(String, String)]) -> usize {
    call.iter()
        .filter(|(k, _)| QUASI_IDENTIFIERS.contains(&k.as_str()))
        .count()
}

fn main() -> anyhow::Result<()> {
    let mut corpus = "datasets/006-phi-egress/exfilbench/sessions.jsonl".to_string();
    let mut ner_model: Option<String> = None;
    let mut out = "results/006-phi-egress/e1e2.json".to_string();
    let mut sessions_eval: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--corpus" => corpus = it.next().unwrap(),
            "--ner-model" => ner_model = it.next(),
            "--out" => out = it.next().unwrap(),
            "--sessions-eval" => sessions_eval = it.next(),
            _ => {}
        }
    }
    let providers = ProviderRegistry::from_profiles(&Config::default().providers)?;
    let ner_provider = ner_model.as_ref().map(|_| providers.get("ollama").or_else(|_| providers.default()).unwrap());

    // results[detector][attack_type_or_"legit"] = (flagged, total)
    let mut res: BTreeMap<String, BTreeMap<String, (u64, u64)>> = BTreeMap::new();
    // Per-session signals for E3/E5/E6 (only emitted when --sessions-eval is set).
    let mut session_records: Vec<Value> = Vec::new();
    for line in std::fs::read_to_string(&corpus)?.lines() {
        if line.trim().is_empty() { continue; }
        let s: Value = serde_json::from_str(line)?;
        let attack = s["attack_type"].as_str().unwrap().to_string();
        let legit = s["legit"].as_bool().unwrap_or(false);
        let args = arg_strings(&s["outbound"]["args"]);

        if sessions_eval.is_some() {
            let calls = session_calls(&s);
            let first_call = calls.first().cloned().unwrap_or_default();
            let mut sess_store = TaintStore::new();
            for r in s["inbound"].as_array().unwrap() {
                sess_store.tag(r["value"].as_str().unwrap(), r["prov_id"].as_str().unwrap(), r["label"].as_str().unwrap());
            }
            let t0 = std::time::Instant::now();
            let _ = sess_store.scan_call(&first_call);
            let taint_us = t0.elapsed().as_secs_f64() * 1e6;
            session_records.push(serde_json::json!({
                "attack_type": attack,
                "legit": legit,
                "external_bytes": external_bytes(&calls),
                "qi_count": qi_count(&first_call),
                "taint_us": taint_us,
            }));
        }

        let mut store = TaintStore::new();
        for r in s["inbound"].as_array().unwrap() {
            store.tag(r["value"].as_str().unwrap(), r["prov_id"].as_str().unwrap(), r["label"].as_str().unwrap());
        }
        let mut detectors: Vec<Box<dyn Detector>> =
            vec![Box::new(RegexDlp::new()), Box::new(TaintDetector { store })];
        if let (Some(p), Some(m)) = (&ner_provider, &ner_model) {
            detectors.push(Box::new(LlmNerDetector { provider: p.clone(), model: m.clone() }));
        }
        for d in &detectors {
            let flagged = !d.scan(&args).is_empty();
            let key = if legit { "legit".to_string() } else { attack.clone() };
            let e = res.entry(d.name().to_string()).or_default().entry(key).or_insert((0, 0));
            e.1 += 1;
            if flagged { e.0 += 1; }
        }
    }
    std::fs::create_dir_all(std::path::Path::new(&out).parent().unwrap())?;
    std::fs::write(&out, serde_json::to_string_pretty(&res)?)?;
    println!("wrote {out}");

    if let Some(path) = sessions_eval {
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&session_records)?)?;
        println!("wrote {path}");
    }
    Ok(())
}
