//! E2 (paper 005): Cedar-vs-Rego decision latency. WARM = engine loaded once,
//! decision only (the fair RQ2 comparison). COLD = parse + decide each time (the
//! shipped RegoEngine reparses per decide; reported separately). Swept over
//! synthetic policy sizes. Build with `--features policy`; on the default build
//! this target prints a notice and exits.
//!
//! Output (stdout JSON) is parsed by scripts/005-policy-authz/experiments.py into
//! results/005-policy-authz/e2_latency/summary.json.

#[cfg(not(feature = "policy"))]
fn main() {
    eprintln!("policy_latency bench requires --features policy");
}

#[cfg(feature = "policy")]
fn main() {
    imp::run();
}

#[cfg(feature = "policy")]
mod imp {
    use qfire::policy::cedar::CedarEngine;
    use qfire::policy::rego::RegoEngine;
    use qfire::policy::{PolicyEngine, Request};
    use serde_json::json;
    use std::time::Instant;

    // A request that matches the prescriber+order_medication path under context.
    fn sample_request() -> Request {
        Request {
            principal: "prescriber".into(),
            action: "order_medication".into(),
            resource: "patient-1".into(),
            args: json!({"drug": "morphine", "dose_mg": 10}),
            attrs: json!({"active_encounter": true, "paneled": true, "max_dose_mg": 30}),
        }
    }

    // Generate a Cedar policy with `n` permit rules (the last matches the request).
    fn cedar_policy(n: usize) -> String {
        let mut s = String::new();
        for i in 0..n.saturating_sub(1) {
            s.push_str(&format!(
                "permit(principal == Role::\"role{i}\", action == Action::\"act{i}\", resource);\n"
            ));
        }
        s.push_str(concat!(
            "permit(principal == Role::\"prescriber\", action == Action::\"order_medication\", resource)\n",
            "when { context.attrs.active_encounter == true && context.attrs.paneled == true && ",
            "context.args.dose_mg <= context.attrs.max_dose_mg };\n",
        ));
        s
    }

    // Generate a Rego policy with `n` allow rules (semantically matching).
    fn rego_policy(n: usize) -> String {
        let mut s = String::from("package authz\n\ndefault allow := false\n\n");
        for i in 0..n.saturating_sub(1) {
            s.push_str(&format!(
                "allow if {{\n    input.principal == \"role{i}\"\n    input.action == \"act{i}\"\n}}\n"
            ));
        }
        s.push_str(concat!(
            "allow if {\n",
            "    input.principal == \"prescriber\"\n",
            "    input.action == \"order_medication\"\n",
            "    input.attrs.active_encounter == true\n",
            "    input.attrs.paneled == true\n",
            "    input.args.dose_mg <= input.attrs.max_dose_mg\n",
            "}\n",
        ));
        s
    }

    fn percentile(sorted_us: &[f64], p: f64) -> f64 {
        if sorted_us.is_empty() {
            return 0.0;
        }
        let idx = ((p / 100.0) * (sorted_us.len() as f64 - 1.0)).round() as usize;
        sorted_us[idx.min(sorted_us.len() - 1)]
    }

    // Time `iters` warm decisions and return p50/p99/mean in microseconds.
    fn time_warm<F: FnMut()>(iters: usize, mut f: F) -> (f64, f64, f64) {
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            f();
            samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        (percentile(&samples, 50.0), percentile(&samples, 99.0), mean)
    }

    pub fn run() {
        let req = sample_request();
        let sizes = [10usize, 100, 1000];
        let iters = 2000;
        let mut points = Vec::new();

        for &n in &sizes {
            // --- Cedar: warm (engine holds parsed PolicySet) ---
            let cedar = CedarEngine::from_src(&cedar_policy(n)).expect("cedar parse");
            let (c_p50, c_p99, c_mean) = time_warm(iters, || {
                let _ = cedar.decide(&req);
            });
            // Cedar cold: parse + decide each iteration.
            let csrc = cedar_policy(n);
            let (cc_p50, cc_p99, _cc_mean) = time_warm(iters.min(200), || {
                let e = CedarEngine::from_src(&csrc).expect("cedar parse");
                let _ = e.decide(&req);
            });

            // --- Rego: warm (build one regorus engine, eval only) ---
            // The shipped RegoEngine reparses per decide(); for the fair warm
            // comparison we drive regorus directly with the engine pre-loaded.
            let rsrc = rego_policy(n);
            let mut warm_engine = regorus::Engine::new();
            warm_engine
                .add_policy("authz.rego".into(), rsrc.clone())
                .expect("rego load");
            let input = regorus::Value::from_json_str(
                &json!({"principal": req.principal, "action": req.action,
                        "resource": req.resource, "args": req.args, "attrs": req.attrs})
                    .to_string(),
            )
            .expect("rego input");
            let (r_p50, r_p99, r_mean) = time_warm(iters, || {
                warm_engine.set_input(input.clone());
                let _ = warm_engine.eval_rule("data.authz.allow".to_string());
            });
            // Rego cold: the shipped per-decide path (reparse each call).
            let rego_cold = RegoEngine::from_src(&rsrc);
            let (rc_p50, rc_p99, _rc_mean) = time_warm(iters.min(200), || {
                let _ = rego_cold.decide(&req);
            });

            points.push(json!({
                "rules": n,
                "cedar_warm_p50_us": c_p50, "cedar_warm_p99_us": c_p99, "cedar_warm_mean_us": c_mean,
                "cedar_cold_p50_us": cc_p50, "cedar_cold_p99_us": cc_p99,
                "rego_warm_p50_us": r_p50, "rego_warm_p99_us": r_p99, "rego_warm_mean_us": r_mean,
                "rego_cold_p50_us": rc_p50, "rego_cold_p99_us": rc_p99,
            }));
        }

        let out = json!({
            "experiment": "E2 Cedar-vs-Rego decision latency",
            "iters": iters,
            "unit": "microseconds",
            "note": "warm = engine pre-loaded, decision only (fair RQ2 comparison). \
                     cold = parse+decide each call (the shipped RegoEngine reparses per decide; \
                     engine-reuse is the scheduled cleanup).",
            "points": points,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    }
}
