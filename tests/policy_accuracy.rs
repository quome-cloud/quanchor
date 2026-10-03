//! E2E (paper 005): decision accuracy over PolicyBench, and Cedar/Rego parity.
//! StaticRbac (context-blind) is the over-permissioning baseline; the context-aware
//! engines (feature-gated) must score higher and agree with each other.

use qfire::policy::{rbac::StaticRbac, Effect, PolicyEngine, Request};
use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn gen_cases(out: &std::path::Path) {
    let s = Command::new("python3")
        .arg(root().join("scripts/005-policy-authz/gen.py"))
        .arg("--out")
        .arg(out)
        .status()
        .expect("gen.py");
    assert!(s.success());
}

fn load_cases(dir: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("cases.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn to_req(c: &serde_json::Value) -> Request {
    Request {
        principal: c["principal"].as_str().unwrap().to_string(),
        action: c["action"].as_str().unwrap().to_string(),
        resource: c["resource"].as_str().unwrap_or("patient-1").to_string(),
        args: c.get("args").cloned().unwrap_or(serde_json::json!({})),
        attrs: c.get("attrs").cloned().unwrap_or(serde_json::json!({})),
    }
}

fn effect_str(e: Effect) -> &'static str {
    match e {
        Effect::Allow => "allow",
        Effect::Deny => "deny",
        Effect::Escalate => "escalate",
    }
}

fn accuracy(eng: &dyn PolicyEngine, cases: &[serde_json::Value]) -> f64 {
    let mut ok = 0;
    for c in cases {
        if effect_str(eng.decide(&to_req(c)).effect) == c["expect"].as_str().unwrap() {
            ok += 1;
        }
    }
    ok as f64 / cases.len() as f64
}

#[test]
fn static_rbac_overpermits_baseline() {
    let dir = tempfile::tempdir().unwrap();
    gen_cases(dir.path());
    let cases = load_cases(dir.path());
    let acc = accuracy(&StaticRbac::clinical_default(), &cases);
    // The paper's E1 finding: context-blind RBAC over-permits badly (~57% on this
    // corpus). Guard the headline with a loose upper bound, not just acc < 1.0.
    assert!(acc < 0.70, "static RBAC should over-permit substantially, got {acc}");
}

#[cfg(feature = "policy")]
#[test]
fn context_engines_accurate_and_agree() {
    use qfire::policy::{cedar::CedarEngine, rego::RegoEngine};
    let dir = tempfile::tempdir().unwrap();
    gen_cases(dir.path());
    let cases = load_cases(dir.path());
    let cedar_src =
        std::fs::read_to_string(root().join("datasets/005-policy-authz/cedar/clinical.cedar"))
            .unwrap();
    let rego_src =
        std::fs::read_to_string(root().join("datasets/005-policy-authz/rego/clinical.rego"))
            .unwrap();
    let cedar = CedarEngine::from_src(&cedar_src).unwrap();
    let rego = RegoEngine::from_src(&rego_src);

    // both context-aware engines should be high-accuracy and strictly beat RBAC
    let rbac_acc = accuracy(&StaticRbac::clinical_default(), &cases);
    let cedar_acc = accuracy(&cedar, &cases);
    let rego_acc = accuracy(&rego, &cases);
    assert!(cedar_acc >= 0.95, "cedar acc {cedar_acc}");
    assert!(rego_acc >= 0.95, "rego acc {rego_acc}");
    assert!(
        cedar_acc > rbac_acc && rego_acc > rbac_acc,
        "context engines must beat RBAC"
    );

    // parity: Cedar and Rego must agree on every case (disagreement is a finding)
    for c in &cases {
        let cd = cedar.decide(&to_req(c)).effect;
        let rg = rego.decide(&to_req(c)).effect;
        assert_eq!(cd, rg, "Cedar/Rego disagree on case {c}");
    }
}
