//! The benchmark harness (`qfire bench`) and attack-corpus adapters.
//!
//! Replays a corpus of attack prompts and a paired corpus of benign in-scope
//! prompts through one or more chains and computes, per rule and per chain:
//! successful-injection rate, block rate, false-positive/negative rates,
//! precision/recall/F1, AUC (from node scores), latency (p50/p95/p99) and the
//! firewall's token/cost overhead. Runs are deterministic and seeded and fully
//! described by a run manifest, so results are reproducible and citable.

mod corpus;
pub(crate) mod metrics;
mod report;

pub use corpus::{load_prompts, Corpus};
pub use metrics::Metrics;

use crate::app::App;
use crate::cli::{exit, AttackCommand};
use crate::harness::engine_hook::MfiHarness;
use crate::harness::mfi::TemplateRewriter;
use crate::harness::provenance::ProvenanceLog;
use crate::ir::LlmRequest;
use crate::verdict::Verdict;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// One evaluated sample: a labeled prompt and the chain's response to it.
#[derive(Clone)]
pub struct Sample {
    pub is_attack: bool,
    pub terminal: Verdict,
    /// The chain's block-score: the maximum node injection/block score in the
    /// trace, used for ROC/AUC.
    pub score: f64,
    pub wall_clock_ms: f64,
    pub summed_detector_ms: f64,
    /// Per-rule verdicts from the trace, for per-rule metrics.
    pub rule_verdicts: HashMap<String, Verdict>,
    /// Optional task identifier for per-task stratification.
    pub task_id: Option<String>,
}

/// The full result of a benchmark run for one chain.
#[derive(Serialize)]
pub struct ChainReport {
    pub chain: String,
    pub chain_version: String,
    pub overall: Metrics,
    pub per_rule: Vec<(String, Metrics)>,
    /// Attack-in-prompt (camouflaged) metrics, when that mode was run.
    pub attack_in_prompt: Option<Metrics>,
    /// Per-task metrics, keyed by `Sample.task_id`. Empty when no task labels.
    pub per_task: std::collections::BTreeMap<String, Metrics>,
    /// Wall-clock seconds to evaluate the whole corpus for this chain.
    pub total_wall_ms: f64,
    /// Prompts per second = corpus size / total wall (load-test throughput).
    pub throughput_qps: f64,
}

/// The run manifest embedded in every artifact for reproducibility.
#[derive(Serialize, Clone)]
pub struct Manifest {
    pub qfire_version: String,
    pub timestamp: String,
    pub seed: u64,
    pub model: String,
    pub chains: Vec<String>,
    pub attack_count: usize,
    pub benign_count: usize,
    pub attack_in_prompt: bool,
}

/// Group samples by `task_id` and compute per-task metrics. Samples with no
/// `task_id` are ignored. Returns an empty map when no sample is task-labeled.
pub(crate) fn group_by_task(samples: &[Sample]) -> std::collections::BTreeMap<String, Metrics> {
    use std::collections::BTreeMap;
    let mut by_task: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
    for s in samples {
        if let Some(t) = &s.task_id {
            by_task.entry(t.clone()).or_default().push(s.clone());
        }
    }
    by_task
        .into_iter()
        .map(|(task, group)| {
            let m = Metrics::from_samples_with_tilt(
                &group,
                |s| s.terminal == crate::verdict::Verdict::Block,
                |s| s.score,
                // tilt = block-score on the attacked split (0 for benign samples)
                |s| if s.is_attack { s.score } else { 0.0 },
            );
            (task, m)
        })
        .collect()
}

/// Run `f` over `items` with up to `concurrency` futures in flight, returning
/// results in the original input order. `concurrency <= 1` runs sequentially.
pub async fn map_concurrent<T, R, Fut, F>(items: Vec<T>, concurrency: usize, f: F) -> Vec<R>
where
    F: Fn(T) -> Fut,
    Fut: std::future::Future<Output = R>,
{
    use futures::stream::StreamExt;
    let n = concurrency.max(1);
    let mut indexed: Vec<(usize, R)> = futures::stream::iter(items.into_iter().enumerate())
        .map(|(i, item)| {
            let fut = f(item);
            async move { (i, fut.await) }
        })
        .buffer_unordered(n)
        .collect()
        .await;
    indexed.sort_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, r)| r).collect()
}

/// Parse a `--task` spec `"name=attacks_path:benign_path"` into its parts.
pub(crate) fn parse_task_spec(spec: &str) -> crate::Result<(String, std::path::PathBuf, std::path::PathBuf)> {
    let (name, paths) = spec
        .split_once('=')
        .ok_or_else(|| crate::Error::Config(format!("--task '{spec}' must be name=attacks:benign")))?;
    let (attacks, benign) = paths
        .rsplit_once(':')
        .ok_or_else(|| crate::Error::Config(format!("--task '{spec}' paths must be attacks:benign")))?;
    if name.is_empty() || attacks.is_empty() || benign.is_empty() {
        return Err(crate::Error::Config(format!("--task '{spec}' has an empty field")));
    }
    Ok((name.to_string(), attacks.into(), benign.into()))
}

/// Parse `--harness` spec `"task=FROM::TO"` into (task, find, replace).
pub(crate) fn parse_harness_spec(spec: &str) -> crate::Result<(String, String, String)> {
    let (task, sub) = spec.split_once('=')
        .ok_or_else(|| crate::Error::Config(format!("--harness '{spec}' must be task=FROM::TO")))?;
    let (from, to) = sub.split_once("::")
        .ok_or_else(|| crate::Error::Config(format!("--harness '{spec}' must contain FROM::TO")))?;
    if task.is_empty() || from.is_empty() {
        return Err(crate::Error::Config(format!("--harness '{spec}' has an empty field")));
    }
    Ok((task.to_string(), from.to_string(), to.to_string()))
}

/// Build an `MfiHarness`-wrapped engine for a single task substitution (from->to).
fn make_harnessed_engine(
    base_providers: &Arc<crate::provider::ProviderRegistry>,
    cache_enabled: bool,
    concurrency: usize,
    from: &str,
    to: &str,
) -> crate::engine::Engine {
    let log = Arc::new(Mutex::new(ProvenanceLog::new()));
    let h = Arc::new(MfiHarness::new(
        TemplateRewriter { subs: vec![(from.to_string(), to.to_string())] },
        "bench-harness",
        log,
    ));
    crate::engine::Engine::new(base_providers.clone())
        .with_cache(cache_enabled)
        .with_concurrency(concurrency)
        .with_harness(h)
}

/// Run the benchmark across the requested chains and write artifacts.
pub async fn run_bench(app: &App, args: &crate::cli::BenchArgs, json: bool) -> crate::Result<()> {
    let mut attacks = load_prompts(&args.attacks)?;
    let mut benign = load_prompts(&args.benign)?;
    if args.limit > 0 {
        attacks.truncate(args.limit);
        benign.truncate(args.limit);
    }
    if attacks.is_empty() && benign.is_empty() && args.tasks.is_empty() {
        return Err(crate::Error::Config(format!(
            "no prompts found under {} or {}",
            args.attacks.display(),
            args.benign.display()
        )));
    }

    let manifest = Manifest {
        qfire_version: crate::VERSION.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        seed: args.seed,
        model: crate::cli::default_model(app),
        chains: args.chains.clone(),
        attack_count: attacks.len(),
        benign_count: benign.len(),
        attack_in_prompt: args.attack_in_prompt,
    };

    if !json {
        eprintln!(
            "bench: {} attacks, {} benign, {} chain(s), seed {}",
            attacks.len(),
            benign.len(),
            args.chains.len(),
            args.seed
        );
    }

    // Build a dedicated engine so the verdict cache can be disabled for honest,
    // un-warmed per-chain latency (and so chains don't share cached verdicts).
    let engine = crate::engine::Engine::new(app.engine.providers().clone())
        .with_cache(!args.no_cache)
        .with_concurrency(args.engine_concurrency);

    // Parse per-task corpora from --task specs.
    let mut task_corpora: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
    for spec in &args.tasks {
        let (name, attacks_path, benign_path) = parse_task_spec(spec)?;
        let mut task_attacks = load_prompts(&attacks_path)?;
        let mut task_benign = load_prompts(&benign_path)?;
        if args.limit > 0 {
            task_attacks.truncate(args.limit);
            task_benign.truncate(args.limit);
        }
        task_corpora.push((name, task_attacks, task_benign));
    }

    // Parse --harness specs into a task -> (from, to) map.
    let mut harness_map: HashMap<String, (String, String)> = HashMap::new();
    for spec in &args.harness {
        let (task, from, to) = parse_harness_spec(spec)?;
        harness_map.insert(task, (from, to));
    }

    let mut reports = Vec::new();
    for chain_name in &args.chains {
        let report = bench_chain(app, &engine, chain_name, &attacks, &benign, &task_corpora, &harness_map, args).await?;
        reports.push(report);
    }

    // Write artifacts.
    std::fs::create_dir_all(&args.out)?;
    report::write_json(&args.out, &manifest, &reports)?;
    report::write_csv(&args.out, &reports)?;
    report::write_markdown(&args.out, &manifest, &reports)?;

    if json {
        let val = serde_json::json!({ "manifest": manifest, "reports": reports });
        println!("{}", serde_json::to_string_pretty(&val)?);
    } else {
        print!("{}", report::render_console(&manifest, &reports));
        eprintln!("\nartifacts written to {}/", args.out.display());
    }
    Ok(())
}

// bench_chain threads several independent benchmark inputs (corpora, args, harness
// map); they have no cohesive sub-grouping, so an allow is clearer than a wrapper.
#[allow(clippy::too_many_arguments)]
async fn bench_chain(
    app: &App,
    engine: &crate::engine::Engine,
    chain_name: &str,
    attacks: &[String],
    benign: &[String],
    task_corpora: &[(String, Vec<String>, Vec<String>)],
    harness_map: &HashMap<String, (String, String)>,
    args: &crate::cli::BenchArgs,
) -> crate::Result<ChainReport> {
    let chain = app.resolve_chain(chain_name)?;
    let compiled = app.compile_for(&chain)?;
    let referenced = chain.referenced_rules()?;

    // Labeled work list in corpus order (attacks then benign), with optional task label.
    // Untagged corpus gets None; per-task corpora get Some(name).
    // --diagonal: only include a task's samples when this chain is matrix_<task>.
    let mut work: Vec<(String, bool, Option<String>)> = Vec::with_capacity(attacks.len() + benign.len());
    work.extend(attacks.iter().map(|p| (p.clone(), true, None)));
    work.extend(benign.iter().map(|p| (p.clone(), false, None)));
    for (name, task_attacks, task_benign) in task_corpora {
        // --diagonal: skip this task unless the current chain is matrix_<task>.
        if args.diagonal && chain_name != format!("matrix_{name}") {
            continue;
        }
        work.extend(task_attacks.iter().map(|p| (p.clone(), true, Some(name.clone()))));
        work.extend(task_benign.iter().map(|p| (p.clone(), false, Some(name.clone()))));
    }

    // Build per-task harnessed engines (one per task that has a harness spec).
    // Tasks without a harness spec use the base engine; the untagged corpus always
    // uses the base engine. This keeps the no-harness/no-monitor path byte-identical.
    let providers = app.engine.providers().clone();
    let cache_enabled = !args.no_cache;
    let concurrency = args.engine_concurrency;
    let task_engines: HashMap<String, crate::engine::Engine> = harness_map
        .iter()
        .map(|(task, (from, to))| {
            let e = make_harnessed_engine(&providers, cache_enabled, concurrency, from, to);
            (task.clone(), e)
        })
        .collect();

    let wall_start = std::time::Instant::now();
    let results: Vec<crate::Result<Sample>> = map_concurrent(
        work,
        args.load_concurrency,
        |(prompt, is_attack, task_id)| {
            let chain_ref = &chain;
            let compiled_ref = &compiled;
            let req = LlmRequest::user("bench", &prompt);
            // Select the engine for this item: harnessed engine if the task has one,
            // otherwise the base engine. The base engine is used for all untagged samples.
            let active_engine: &crate::engine::Engine = task_id
                .as_deref()
                .and_then(|t| task_engines.get(t))
                .unwrap_or(engine);
            let monitor_output = args.monitor_output;
            let prov_clone = providers.clone();
            async move {
                let decision = active_engine.evaluate(chain_ref, compiled_ref, &req).await?;
                // --monitor-output: when pre-forward decision is ALLOW, ask the
                // provider for a response and re-evaluate the output layer.
                if monitor_output && decision.terminal == Verdict::Allow {
                    let resp_text = match prov_clone.default() {
                        Ok(prov) => match prov.complete(&req).await {
                            Ok(r) => r.content,
                            Err(_) => {
                                // Provider error: fall back to pre-forward decision.
                                let mut sample = sample_from(&decision, is_attack);
                                sample.task_id = task_id;
                                return Ok(sample);
                            }
                        },
                        Err(_) => {
                            let mut sample = sample_from(&decision, is_attack);
                            sample.task_id = task_id;
                            return Ok(sample);
                        }
                    };
                    let out_decision = match active_engine
                        .evaluate_output(chain_ref, compiled_ref, &req, &resp_text)
                        .await
                    {
                        Ok(d) => d,
                        Err(_) => decision, // post-response eval failed; keep the pre-forward decision
                    };
                    let mut sample = sample_from(&out_decision, is_attack);
                    sample.task_id = task_id;
                    Ok(sample)
                } else {
                    let mut sample = sample_from(&decision, is_attack);
                    sample.task_id = task_id;
                    Ok(sample)
                }
            }
        },
    )
    .await;
    let total_wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let samples: Vec<Sample> = results.into_iter().collect::<crate::Result<Vec<_>>>()?;
    let throughput_qps = if total_wall_ms > 0.0 {
        samples.len() as f64 / (total_wall_ms / 1000.0)
    } else {
        0.0
    };

    // Optional per-prompt prediction dump (corpus order: attacks then benign),
    // for paired tests (McNemar) and bootstrap CIs across chains.
    if let Some(dir) = &args.dump {
        std::fs::create_dir_all(dir)?;
        use std::io::Write as _;
        let mut f = std::fs::File::create(dir.join(format!("{}.jsonl", chain.id)))?;
        for s in &samples {
            writeln!(
                f,
                "{}",
                serde_json::json!({
                    "is_attack": s.is_attack,
                    "blocked": s.terminal == Verdict::Block,
                    "score": s.score
                })
            )?;
        }
    }

    let overall = Metrics::from_samples(&samples, |s| s.terminal == Verdict::Block, |s| s.score);
    let per_task = group_by_task(&samples);
    let per_rule = referenced
        .iter()
        .map(|rid| {
            let rid2 = rid.clone();
            let m = Metrics::from_samples(
                &samples,
                move |s| s.rule_verdicts.get(&rid2) == Some(&Verdict::Block),
                |s| s.score,
            );
            (rid.clone(), m)
        })
        .collect();

    // Attack-in-prompt: camouflage payloads inside benign prompts.
    let attack_in_prompt = if args.attack_in_prompt {
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
        let mutated = corpus::attack_in_prompt(benign, &mut rng);
        let mut aip_samples = Vec::new();
        for p in &mutated {
            let req = LlmRequest::user("bench", p);
            let decision = engine.evaluate(&chain, &compiled, &req).await?;
            aip_samples.push(sample_from(&decision, true));
        }
        Some(Metrics::from_samples(
            &aip_samples,
            |s| s.terminal == Verdict::Block,
            |s| s.score,
        ))
    } else {
        None
    };

    Ok(ChainReport {
        chain: chain.id.clone(),
        chain_version: chain.version.clone(),
        overall,
        per_rule,
        attack_in_prompt,
        per_task,
        total_wall_ms,
        throughput_qps,
    })
}

fn sample_from(decision: &crate::engine::Decision, is_attack: bool) -> Sample {
    // Coherent chain block-score for ROC/AUC: every node contributes a value in
    // [0,1] -- the calibrated injection probability for scoring detectors
    // (deberta/judge/similarity), or the block-confidence for lexical blockers
    // (regex/aho/entropy), which contribute only when they actually BLOCK. This
    // avoids mixing raw entropy bits (a different scale) into the ranking score,
    // which previously inverted the AUC for multi-detector chains.
    let score = decision
        .trace
        .rules
        .iter()
        .flat_map(|r| r.nodes.iter())
        .map(|n| match n.kind.as_str() {
            "deberta" | "judge" | "similarity" | "goal_drift" | "output_monitor" => n.score.unwrap_or(n.confidence),
            _ => {
                if n.verdict == Verdict::Block {
                    n.confidence
                } else {
                    0.0
                }
            }
        })
        .fold(0.0_f64, f64::max);
    let rule_verdicts = decision
        .trace
        .rules
        .iter()
        .map(|r| (r.rule_id.clone(), r.verdict))
        .collect();
    Sample {
        is_attack,
        terminal: decision.terminal,
        score,
        wall_clock_ms: decision.trace.wall_clock_ms,
        summed_detector_ms: decision.trace.summed_detector_ms,
        rule_verdicts,
        task_id: None,
    }
}

#[cfg(test)]
mod parse_harness_spec_tests {
    use super::parse_harness_spec;

    #[test]
    fn parse_harness_spec_ok() {
        let (task, from, to) = parse_harness_spec("dose=safe range::danger zone").unwrap();
        assert_eq!(task, "dose");
        assert_eq!(from, "safe range");
        assert_eq!(to, "danger zone");
    }

    #[test]
    fn parse_harness_spec_ok_empty_to() {
        // `to` may be empty (erase a phrase entirely)
        let (task, from, to) = parse_harness_spec("dose=safe range::").unwrap();
        assert_eq!(task, "dose");
        assert_eq!(from, "safe range");
        assert_eq!(to, "");
    }

    #[test]
    fn parse_harness_spec_rejects_no_equals() {
        assert!(parse_harness_spec("noseparator").is_err());
    }

    #[test]
    fn parse_harness_spec_rejects_no_double_colon() {
        assert!(parse_harness_spec("dose=onlyone").is_err());
    }

    #[test]
    fn parse_harness_spec_rejects_empty_task() {
        assert!(parse_harness_spec("=from::to").is_err());
    }

    #[test]
    fn parse_harness_spec_rejects_empty_from() {
        assert!(parse_harness_spec("dose=::to").is_err());
    }
}

#[cfg(test)]
mod parse_task_spec_tests {
    use super::parse_task_spec;

    #[test]
    fn parse_task_spec_ok() {
        let (n, a, b) = parse_task_spec("dose=datasets/m/dose/attacks.jsonl:datasets/m/dose/benign.jsonl").unwrap();
        assert_eq!(n, "dose");
        assert_eq!(a, std::path::PathBuf::from("datasets/m/dose/attacks.jsonl"));
        assert_eq!(b, std::path::PathBuf::from("datasets/m/dose/benign.jsonl"));
    }

    #[test]
    fn parse_task_spec_rejects_malformed() {
        assert!(parse_task_spec("noequals").is_err());
        assert!(parse_task_spec("name=onlyone").is_err());
        assert!(parse_task_spec("=a:b").is_err());
    }
}

#[cfg(test)]
mod group_by_task_tests {
    use super::{group_by_task, Sample};

    #[test]
    fn group_by_task_splits_and_counts() {
        use crate::verdict::Verdict;
        let mk = |task: &str, attack: bool, v: Verdict, score: f64| Sample {
            is_attack: attack,
            terminal: v,
            score,
            wall_clock_ms: 0.0,
            summed_detector_ms: 0.0,
            rule_verdicts: Default::default(),
            task_id: Some(task.into()),
        };
        let samples = vec![
            mk("dose", true, Verdict::Block, 0.9),
            mk("dose", false, Verdict::Allow, 0.1),
            mk("triage", true, Verdict::Allow, 0.4),
        ];
        let g = group_by_task(&samples);
        assert_eq!(g.len(), 2);
        assert!(g.contains_key("dose"));
        assert!(g.contains_key("triage"));
        // dose: 1 attack blocked -> recall 1.0; triage: 1 attack allowed -> recall 0.0
        assert!((g["dose"].recall - 1.0).abs() < 1e-9);
        assert!((g["triage"].recall - 0.0).abs() < 1e-9);
    }

    #[test]
    fn group_by_task_ignores_unlabeled() {
        use crate::verdict::Verdict;
        let s = vec![Sample {
            is_attack: true,
            terminal: Verdict::Block,
            score: 0.9,
            wall_clock_ms: 0.0,
            summed_detector_ms: 0.0,
            rule_verdicts: Default::default(),
            task_id: None,
        }];
        assert!(group_by_task(&s).is_empty());
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::map_concurrent;

    #[tokio::test]
    async fn preserves_order_and_count() {
        let out = map_concurrent(vec![1, 2, 3, 4, 5], 8, |x| async move { x * 2 }).await;
        assert_eq!(out, vec![2, 4, 6, 8, 10]); // order preserved despite concurrency
        assert_eq!(out.len(), 5);
    }

    #[tokio::test]
    async fn concurrency_does_not_change_results() {
        let seq = map_concurrent(vec![1, 2, 3, 4, 5], 1, |x| async move { x * 2 }).await;
        let par = map_concurrent(vec![1, 2, 3, 4, 5], 8, |x| async move { x * 2 }).await;
        assert_eq!(seq, par);
    }
}

/// `qfire attack` subcommands.
pub async fn run_attack(cmd: AttackCommand, json: bool) -> crate::Result<i32> {
    match cmd {
        AttackCommand::Import(args) => {
            let prompts = corpus::import(&args.source, &args.format)?;
            corpus::write_jsonl(&args.out, &prompts, &args.source.display().to_string())?;
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "imported": prompts.len(), "out": args.out.display().to_string() })
                );
            } else {
                println!("imported {} prompts → {}", prompts.len(), args.out.display());
            }
            Ok(exit::ALLOW)
        }
        AttackCommand::Mutate(args) => {
            let benign = load_prompts(&args.benign)?;
            let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
            let mutated = corpus::attack_in_prompt(&benign, &mut rng);
            corpus::write_jsonl(&args.out, &mutated, "attack-in-prompt")?;
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "mutated": mutated.len(), "out": args.out.display().to_string() })
                );
            } else {
                println!("wrote {} attack-in-prompt cases → {}", mutated.len(), args.out.display());
            }
            Ok(exit::ALLOW)
        }
    }
}
