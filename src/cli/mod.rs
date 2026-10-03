//! The QFIRE command-line interface.
//!
//! Follows cargo/kubectl conventions: subcommands, aligned columnar human output
//! by default, a `--json` flag on every command for machine-readable output, and
//! a `--quiet` flag for CI. Exit codes are meaningful so QFIRE can gate
//! pipelines: `0` = allowed, `2` = blocked, `1` = error.

mod rules_cmd;

use crate::app::App;
use crate::ir::LlmRequest;
use crate::output;
use crate::verdict::Verdict;
use clap::{Args, Parser, Subcommand};
use std::io::Read;
use std::path::PathBuf;

/// Exit codes used throughout the CLI.
pub mod exit {
    pub const ALLOW: i32 = 0;
    pub const ERROR: i32 = 1;
    pub const BLOCK: i32 = 2;
}

#[derive(Parser)]
#[command(
    name = "qfire",
    version,
    about = "QFIRE — a prompt firewall for LLM applications",
    long_about = "QFIRE evaluates inbound prompts against declarative firewall rules and \
                  forwards to a downstream provider only on ALLOW. Surfaces: a proxy port, \
                  this CLI, structured output, and benchmark report artifacts."
)]
pub struct Cli {
    /// Path to a config file (default: ./qfire.toml or ~/.config/qfire/config.toml).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Emit machine-readable JSON instead of human output.
    #[arg(long, global = true)]
    pub json: bool,

    /// Suppress non-essential output (for CI).
    #[arg(long, global = true)]
    pub quiet: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run the proxy daemon with wire-compatible provider endpoints.
    Serve(ServeArgs),
    /// Evaluate a prompt against a chain and print the verdict (no downstream call).
    Check(CheckArgs),
    /// Evaluate and, if allowed, execute the downstream call.
    Run(RunArgs),
    /// Manage the rule library (list, lint, test, explain).
    #[command(subcommand)]
    Rules(rules_cmd::RulesCommand),
    /// Replay an attack corpus through chains and emit research metrics.
    Bench(BenchArgs),
    /// Import or mutate attack corpora (garak / PyRIT adapters).
    #[command(subcommand)]
    Attack(AttackCommand),
    /// Summarize an audit log.
    Report(ReportArgs),
    /// Tamper-evident audit log: verify, prove, head.
    #[command(subcommand)]
    Audit(AuditCommand),
    /// Supply-chain AIBOM: generate, verify, scan.
    #[command(subcommand)]
    Aibom(AibomCommand),
    /// Policy-as-code authorization: decide, test, coverage.
    #[command(subcommand)]
    Policy(PolicyCommand),
    /// Multi-agent identity layer: replay MABench scenarios and report per-attack
    /// success rates and the Handoff Coverage Rate (HCR).
    #[command(subcommand)]
    Identity(IdentityCommand),
    /// Retrieval poison defense: query a corpus, scan for poison, test a memory write.
    #[command(subcommand)]
    Retrieval(RetrievalCommand),
    /// Drift monitor: replay a decision stream off-hot-path and emit graduated alerts.
    Monitor(MonitorArgs),
    /// Regulatory passport (lifecycle): classify risk/autonomy, harmonize across
    /// jurisdictions, verify a passport, and run the admission deployment gate.
    #[command(subcommand)]
    Lifecycle(LifecycleCommand),
}

#[derive(Args)]
pub struct PromptInput {
    /// The prompt text. Use `-` to read from stdin, or omit and use --file.
    pub prompt: Option<String>,
    /// Read the prompt from a file.
    #[arg(long)]
    pub file: Option<PathBuf>,
    /// An optional system prompt.
    #[arg(long)]
    pub system: Option<String>,
    /// The downstream model name (defaults to the profile's model).
    #[arg(long)]
    pub model: Option<String>,
}

impl PromptInput {
    fn into_request(self, default_model: &str) -> crate::Result<LlmRequest> {
        let text = if let Some(path) = &self.file {
            std::fs::read_to_string(path)?
        } else {
            match self.prompt.as_deref() {
                Some("-") | None => {
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s
                }
                Some(p) => p.to_string(),
            }
        };
        let model = self.model.unwrap_or_else(|| default_model.to_string());
        let mut req = LlmRequest::user(model, text.trim_end());
        req.system = self.system;
        Ok(req)
    }
}

#[derive(Args)]
pub struct CheckArgs {
    #[command(flatten)]
    pub input: PromptInput,
    /// The chain (or rule) to evaluate against.
    #[arg(long, short = 'c', default_value = "default")]
    pub chain: String,
}

#[derive(Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub input: PromptInput,
    #[arg(long, short = 'c', default_value = "default")]
    pub chain: String,
    /// Override the downstream provider profile.
    #[arg(long)]
    pub provider: Option<String>,
}

#[derive(Args)]
pub struct ServeArgs {
    /// Address to bind, e.g. 127.0.0.1:8787.
    #[arg(long, default_value = "127.0.0.1:8787")]
    pub addr: String,
    /// The default chain applied when a request selects none.
    #[arg(long, default_value = "default")]
    pub chain: String,
    /// Redact block reasons in the refusal envelope returned to clients.
    #[arg(long)]
    pub redact: bool,
    /// On BLOCK, return a 200 OpenAI-shaped refusal completion for OpenAI-family
    /// requests instead of the default 403 firewall envelope. Lets OpenAI-SDK
    /// clients (agent benchmarks) treat a block as a refusal. Other families and
    /// non-OpenAI routes still get the 403 envelope.
    #[arg(long, default_value_t = false)]
    pub openai_block_refusal: bool,
}

#[derive(Args)]
pub struct BenchArgs {
    /// One or more chains to benchmark.
    #[arg(long = "chain", short = 'c', required = true)]
    pub chains: Vec<String>,
    /// Directory of attack prompts (one per line in .txt, or .jsonl with `prompt`).
    #[arg(long, default_value = "datasets/001-qfire/attacks")]
    pub attacks: PathBuf,
    /// Directory of benign in-scope prompts.
    #[arg(long, default_value = "datasets/001-qfire/benign")]
    pub benign: PathBuf,
    /// Output directory for CSV/JSON/Markdown artifacts.
    #[arg(long, default_value = "results/001-qfire")]
    pub out: PathBuf,
    /// Random seed for deterministic runs.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Also run attack-in-prompt (camouflaged) mutations of benign prompts.
    #[arg(long)]
    pub attack_in_prompt: bool,
    /// Limit prompts per corpus (0 = all).
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    /// Disable the verdict cache (honest, un-warmed per-chain latency).
    #[arg(long)]
    pub no_cache: bool,
    /// Dump per-prompt predictions (one JSONL file per chain) into this directory,
    /// for paired statistics (McNemar, bootstrap CIs).
    #[arg(long)]
    pub dump: Option<PathBuf>,
    /// Max concurrently-running detector nodes (engine semaphore). Default 16.
    #[arg(long, default_value_t = 16)]
    pub engine_concurrency: usize,
    /// Number of prompt evaluations in flight (in-process load test). 1 = sequential.
    #[arg(long, default_value_t = 1)]
    pub load_concurrency: usize,
    /// Per-task corpora for the generality matrix, repeatable:
    /// `--task <name>=<attacks_path>:<benign_path>`. Each labels its samples
    /// with task_id=<name>. When omitted, the run uses only --attacks/--benign.
    #[arg(long = "task")]
    pub tasks: Vec<String>,
    /// Install an MFI harness for a task so the operative MIIM is tilted for the
    /// whole run (exercises goal_drift): repeatable `--harness task=FROM::TO`.
    #[arg(long = "harness")]
    pub harness: Vec<String>,
    /// After a pre-forward ALLOW, run the post-response pass so output_monitor
    /// sees the provider response (exercises the behavioral layer).
    #[arg(long)]
    pub monitor_output: bool,
    /// Run each --task corpus only through its paired chain `matrix_<task>`.
    #[arg(long)]
    pub diagonal: bool,
}

#[derive(Subcommand)]
pub enum AttackCommand {
    /// Import a corpus from garak/PyRIT output or a labeled file into datasets/.
    Import(AttackImportArgs),
    /// Mutate benign prompts into attack-in-prompt variants (PyRIT-style).
    Mutate(AttackMutateArgs),
}

#[derive(Args)]
pub struct AttackImportArgs {
    /// Source file (garak .jsonl report, PyRIT export, or a .txt of prompts).
    pub source: PathBuf,
    /// Source format.
    #[arg(long, default_value = "auto")]
    pub format: String,
    /// Destination file under datasets/.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args)]
pub struct AttackMutateArgs {
    /// Benign prompts file to camouflage attacks inside.
    pub benign: PathBuf,
    /// Output file of mutated attack-in-prompt cases.
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

#[derive(Args)]
pub struct ReportArgs {
    /// Path to the audit log (JSONL).
    #[arg(default_value = "audit.jsonl")]
    pub path: PathBuf,
}

#[derive(Subcommand)]
pub enum AuditCommand {
    /// Verify chain, signatures and (optionally) anchors; exit 2 if tampered.
    Verify(AuditVerifyArgs),
    /// Print a Merkle inclusion proof for one entry.
    Prove(AuditProveArgs),
    /// Print the chain head and last anchor.
    Head(AuditHeadArgs),
}

#[derive(Args)]
pub struct AuditVerifyArgs {
    /// Path to the chained audit log.
    #[arg(long, default_value = "audit.jsonl")]
    pub log: PathBuf,
    /// Path to the anchors JSONL (enables TA6 rollback checking).
    #[arg(long)]
    pub anchors: Option<PathBuf>,
    /// Override the verifying pubkey (hex) instead of the header's.
    #[arg(long)]
    pub pubkey: Option<String>,
}

#[derive(Args)]
pub struct AuditProveArgs {
    #[arg(long, default_value = "audit.jsonl")]
    pub log: PathBuf,
    /// Sequence number to prove inclusion for.
    #[arg(long)]
    pub seq: u64,
}

#[derive(Args)]
pub struct AuditHeadArgs {
    #[arg(long, default_value = "audit.jsonl")]
    pub log: PathBuf,
    #[arg(long)]
    pub anchors: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum AibomCommand {
    /// Enumerate components and write a signed CycloneDX AIBOM.
    Generate(AibomGenerateArgs),
    /// Verify an AIBOM's signature and per-component digests.
    Verify(AibomVerifyArgs),
    /// Scan components against a pinned OSV snapshot.
    Scan(AibomScanArgs),
}

#[derive(Args)]
pub struct AibomGenerateArgs {
    #[arg(long, default_value = "rules")] pub rules: PathBuf,
    #[arg(long, default_value = "chains")] pub chains: PathBuf,
    #[arg(long)] pub config: Option<PathBuf>,
    #[arg(long, default_value = "Cargo.lock")] pub cargo_lock: PathBuf,
    #[arg(long)] pub onnx: Option<PathBuf>,
    #[arg(long, default_value = "aibom.key")] pub key: PathBuf,
    #[arg(long, default_value = "aibom.json")] pub out: PathBuf,
}

#[derive(Args)]
pub struct AibomVerifyArgs {
    #[arg(long, default_value = "aibom.json")] pub aibom: PathBuf,
    #[arg(long)] pub pubkey: Option<String>,
}

#[derive(Args)]
pub struct AibomScanArgs {
    #[arg(long, default_value = "aibom.json")] pub aibom: PathBuf,
    #[arg(long)] pub osv: PathBuf,
}

#[derive(Subcommand)]
pub enum PolicyCommand {
    /// Decide a single request against a policy bundle.
    Decide(PolicyDecideArgs),
    /// Run a labeled request->decision case file and report accuracy.
    Test(PolicyTestArgs),
    /// Print the policy->requirement coverage map.
    Coverage(PolicyCoverageArgs),
}

#[derive(Args)]
pub struct PolicyDecideArgs {
    #[arg(long, default_value = "static_rbac")] pub engine: String,
    #[arg(long)] pub principal: String,
    #[arg(long)] pub action: String,
    #[arg(long, default_value = "patient-1")] pub resource: String,
    /// JSON object of tool arguments.
    #[arg(long, default_value = "{}")] pub args: String,
    /// Path to a policy source file (.cedar or .rego); required for cedar/rego engines.
    #[arg(long)] pub policy: Option<PathBuf>,
}

#[derive(Args)]
pub struct PolicyTestArgs {
    #[arg(long, default_value = "static_rbac")] pub engine: String,
    #[arg(long)] pub cases: PathBuf,
    #[arg(long)] pub policy: Option<PathBuf>,
}

#[derive(Args)]
pub struct PolicyCoverageArgs {
    #[arg(long)] pub map: PathBuf,
}

#[derive(Subcommand)]
pub enum IdentityCommand {
    /// Replay a MABench scenario JSONL through the identity layer and report
    /// per-attack success rates plus the Handoff Coverage Rate (HCR).
    Test(IdentityTestArgs),
}

#[derive(Args)]
pub struct IdentityTestArgs {
    /// Path to the MABench scenario JSONL file.
    #[arg(long)]
    pub scenarios: PathBuf,
    /// Defense level: none | jwt | biscuit | biscuit_prov | full
    #[arg(long, default_value = "full")]
    pub defense: String,
}

/// Parse args and dispatch, returning a process exit code.
pub async fn run() -> i32 {
    let cli = Cli::parse();
    match dispatch(cli).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            exit::ERROR
        }
    }
}

async fn dispatch(cli: Cli) -> crate::Result<i32> {
    let color = !cli.quiet && output::use_color();
    match cli.command {
        Command::Check(args) => {
            let app = App::load(cli.config.as_deref())?;
            let model = default_model(&app);
            let req = args.input.into_request(&model)?;
            let decision = app.check(&args.chain, &req).await?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&decision)?);
            } else if cli.quiet {
                println!("{}", decision.terminal.label());
            } else {
                print!("{}", output::render_decision(&decision, color));
            }
            Ok(exit_for(decision.terminal))
        }
        Command::Run(args) => {
            let app = App::load(cli.config.as_deref())?;
            let model = default_model(&app);
            let req = args.input.into_request(&model)?;
            let (decision, response) = app.run(&args.chain, &req, args.provider.as_deref()).await?;
            if cli.json {
                let val = serde_json::json!({ "decision": decision, "response": response });
                println!("{}", serde_json::to_string_pretty(&val)?);
            } else {
                if !cli.quiet {
                    print!("{}", output::render_decision(&decision, color));
                    println!("{}", "─".repeat(60));
                }
                match response {
                    Some(r) => {
                        println!("{}", r.content);
                        if !cli.quiet {
                            eprintln!(
                                "[{} tokens, ${:.6}]",
                                r.usage.total_tokens(),
                                r.usage.cost_usd
                            );
                        }
                    }
                    None => {
                        if !cli.quiet {
                            eprintln!("(blocked — downstream not contacted)");
                        }
                    }
                }
            }
            Ok(exit_for(decision.terminal))
        }
        Command::Rules(cmd) => {
            rules_cmd::run(cmd, cli.config.as_deref(), cli.json, cli.quiet).await
        }
        Command::Serve(args) => {
            let app = App::load(cli.config.as_deref())?;
            crate::proxy::serve(
                app,
                &args.addr,
                &args.chain,
                args.redact,
                args.openai_block_refusal,
            )
            .await?;
            Ok(exit::ALLOW)
        }
        Command::Bench(args) => {
            let app = App::load(cli.config.as_deref())?;
            crate::bench::run_bench(&app, &args, cli.json).await?;
            Ok(exit::ALLOW)
        }
        Command::Attack(cmd) => crate::bench::run_attack(cmd, cli.json).await,
        Command::Report(args) => report(&args, cli.json),
        Command::Audit(cmd) => run_audit(cmd),
        Command::Aibom(cmd) => run_aibom(cmd),
        Command::Policy(cmd) => run_policy(cmd),
        Command::Identity(cmd) => run_identity(cmd),
        Command::Retrieval(cmd) => run_retrieval(cmd),
        Command::Monitor(args) => run_monitor(args, cli.json),
        Command::Lifecycle(cmd) => run_lifecycle(cmd, cli.json),
    }
}

fn exit_for(v: Verdict) -> i32 {
    match v {
        Verdict::Allow => exit::ALLOW,
        Verdict::Block => exit::BLOCK,
        _ => exit::ERROR,
    }
}

/// The default model name from the default provider profile.
pub fn default_model(app: &App) -> String {
    app.config
        .providers
        .first()
        .and_then(|p| p.model.clone())
        .unwrap_or_else(|| "llama3.2".to_string())
}

fn run_audit(cmd: AuditCommand) -> crate::Result<i32> {
    use crate::audit::{anchor_sink::FileAnchorSink, chain, merkle, verify};
    match cmd {
        AuditCommand::Verify(a) => {
            let anchors = match &a.anchors {
                Some(p) => Some(FileAnchorSink::read_all(p)?),
                None => None,
            };
            let r = verify::verify_log(&a.log, anchors.as_deref(), a.pubkey.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if !r.ok {
                return Ok(exit::BLOCK);
            }
            Ok(exit::ALLOW)
        }
        AuditCommand::Prove(a) => {
            let text = std::fs::read_to_string(&a.log)?;
            let mut hashes = Vec::new();
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                hashes.push(chain::parse_line(line)?.this_hash);
            }
            let idx = a.seq as usize;
            let proof = merkle::inclusion_proof(&hashes, idx).ok_or_else(|| {
                anyhow::anyhow!("seq {} out of range ({} entries)", a.seq, hashes.len())
            })?;
            let root = merkle::merkle_root(&hashes);
            println!(
                "{}",
                serde_json::json!({
                    "seq": a.seq,
                    "this_hash": hashes[idx],
                    "merkle_root": root,
                    "proof": proof.iter().map(|(h, l)| serde_json::json!({"sibling": h, "sibling_is_left": l})).collect::<Vec<_>>(),
                    "verified": merkle::verify_inclusion(&hashes[idx], &proof, &root),
                })
            );
            Ok(exit::ALLOW)
        }
        AuditCommand::Head(a) => {
            let text = std::fs::read_to_string(&a.log)?;
            let last = text.lines().filter(|l| !l.trim().is_empty()).next_back();
            let head = match last {
                Some(line) => {
                    let e = chain::parse_line(line)?;
                    serde_json::json!({"seq": e.seq, "head": e.this_hash})
                }
                None => serde_json::json!({"seq": null, "head": "GENESIS"}),
            };
            let anchor = match &a.anchors {
                Some(p) => FileAnchorSink::read_all(p)?.last().cloned(),
                None => None,
            };
            println!("{}", serde_json::json!({"chain": head, "last_anchor": anchor}));
            Ok(exit::ALLOW)
        }
    }
}

fn run_aibom(cmd: AibomCommand) -> crate::Result<i32> {
    use crate::admission::aibom::Aibom;
    use crate::admission::attest::{sign_aibom, tampered_components, verify_aibom_sig, SignedAibom};
    use crate::admission::vuln::{OsvSnapshot, VulnFeed};
    use crate::audit::sign::AuditSigner;
    match cmd {
        AibomCommand::Generate(a) => {
            let cargo = a.cargo_lock.exists().then_some(a.cargo_lock.as_path());
            let onnx = a.onnx.as_deref().filter(|p| p.exists());
            let aibom = Aibom::enumerate(&a.rules, &a.chains, a.config.as_deref(), onnx, cargo, &[]);
            let signer = AuditSigner::load_or_generate(&a.key)?;
            let signed = sign_aibom(&aibom, &signer);
            std::fs::write(&a.out, serde_json::to_string_pretty(&signed)?)?;
            println!("{}", serde_json::json!({
                "components": aibom.components.len(),
                "provenance_gap": aibom.provenance_gap(),
                "digest": aibom.document_digest(),
                "out": a.out.display().to_string(),
            }));
            Ok(exit::ALLOW)
        }
        AibomCommand::Verify(a) => {
            let signed: SignedAibom = serde_json::from_str(&std::fs::read_to_string(&a.aibom)?)?;
            let sig_ok = verify_aibom_sig(&signed, a.pubkey.as_deref());
            let aibom = Aibom::from_cyclonedx(&signed.document)?;
            let tampered = tampered_components(&aibom);
            let ok = sig_ok && tampered.is_empty();
            println!("{}", serde_json::json!({
                "signature_ok": sig_ok,
                "tampered": tampered.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
                "ok": ok,
            }));
            Ok(if ok { exit::ALLOW } else { exit::BLOCK })
        }
        AibomCommand::Scan(a) => {
            let signed: SignedAibom = serde_json::from_str(&std::fs::read_to_string(&a.aibom)?)?;
            let aibom = Aibom::from_cyclonedx(&signed.document)?;
            let matches = OsvSnapshot::load(&a.osv)?.matches(&aibom);
            println!("{}", serde_json::to_string_pretty(&matches)?);
            Ok(if matches.is_empty() { exit::ALLOW } else { exit::BLOCK })
        }
    }
}

fn run_policy(cmd: PolicyCommand) -> crate::Result<i32> {
    use crate::policy::{rbac::StaticRbac, Effect, PolicyEngine, Request};
    // Build an engine by name. static_rbac needs no policy file; cedar/rego do
    // and are only available with the `policy` feature.
    fn engine_for(name: &str, _policy: Option<&std::path::Path>) -> crate::Result<Box<dyn PolicyEngine>> {
        match name {
            "static_rbac" => Ok(Box::new(StaticRbac::clinical_default())),
            #[cfg(feature = "policy")]
            "cedar" => {
                let src = std::fs::read_to_string(_policy.ok_or_else(|| crate::error::Error::Config("cedar needs --policy".into()))?)?;
                Ok(Box::new(crate::policy::cedar::CedarEngine::from_src(&src)?))
            }
            #[cfg(feature = "policy")]
            "rego" => {
                let src = std::fs::read_to_string(_policy.ok_or_else(|| crate::error::Error::Config("rego needs --policy".into()))?)?;
                Ok(Box::new(crate::policy::rego::RegoEngine::from_src(&src)))
            }
            other => Err(crate::error::Error::Config(format!("unknown/unavailable engine '{other}' (build with --features policy for cedar/rego)"))),
        }
    }

    match cmd {
        PolicyCommand::Decide(a) => {
            let eng = engine_for(&a.engine, a.policy.as_deref())?;
            let args: serde_json::Value = serde_json::from_str(&a.args)?;
            let req = Request { principal: a.principal, action: a.action, resource: a.resource,
                args, attrs: serde_json::json!({}) };
            let d = eng.decide(&req);
            println!("{}", serde_json::to_string_pretty(&d)?);
            Ok(if d.effect == Effect::Allow { exit::ALLOW } else { exit::BLOCK })
        }
        PolicyCommand::Test(a) => {
            let eng = engine_for(&a.engine, a.policy.as_deref())?;
            let text = std::fs::read_to_string(&a.cases)?;
            // Accuracy plus the OPR/FBR breakdown (paper 005 E1): over_permit =
            // got allow but expected deny; false_block = got deny/escalate but
            // expected allow.
            let (mut correct, mut total) = (0u32, 0u32);
            let (mut over_permit, mut false_block) = (0u32, 0u32);
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let case: serde_json::Value = serde_json::from_str(line)?;
                let req = Request {
                    principal: case["principal"].as_str().unwrap_or("").to_string(),
                    action: case["action"].as_str().unwrap_or("").to_string(),
                    resource: case["resource"].as_str().unwrap_or("patient-1").to_string(),
                    args: case.get("args").cloned().unwrap_or(serde_json::json!({})),
                    attrs: case.get("attrs").cloned().unwrap_or(serde_json::json!({})),
                };
                let got = format!("{:?}", eng.decide(&req).effect).to_lowercase();
                let want = case["expect"].as_str().unwrap_or("");
                total += 1;
                if got == want {
                    correct += 1;
                } else if got == "allow" {
                    over_permit += 1;
                } else {
                    false_block += 1;
                }
            }
            let f = |n: u32| if total > 0 { n as f64 / total as f64 } else { 0.0 };
            println!("{}", serde_json::json!({"engine": a.engine, "correct": correct, "total": total,
                "accuracy": f(correct), "over_permit": over_permit, "false_block": false_block,
                "opr": f(over_permit), "fbr": f(false_block)}));
            Ok(exit::ALLOW)
        }
        PolicyCommand::Coverage(a) => {
            let map: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a.map)?)?;
            println!("{}", serde_json::to_string_pretty(&map)?);
            Ok(exit::ALLOW)
        }
    }
}

// ─── Paper 008: identity test ────────────────────────────────────────────────

fn run_identity(cmd: IdentityCommand) -> crate::Result<i32> {
    match cmd {
        IdentityCommand::Test(a) => {
            use crate::identity::IdentityEffect;
            use std::collections::HashMap;

            let text = std::fs::read_to_string(&a.scenarios)?;
            let defense = a.defense.as_str();

            let mut total = 0u32;
            let mut correct = 0u32;
            let mut attack_success = 0u32; // non-legit scenarios that ended Allow
            let mut false_block = 0u32;    // legit scenarios that ended Deny
            let mut legit_total = 0u32;
            let mut legit_allowed = 0u32;

            // per_attack: attack -> (blocked, count)
            let mut per_attack: HashMap<String, (u32, u32)> = HashMap::new();

            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let sc: serde_json::Value = serde_json::from_str(line)?;
                let attack = sc["attack"].as_str().unwrap_or("unknown").to_string();

                // The expected final decision is the last element of the "expected" array.
                let expected_str = sc["expected"]
                    .as_array()
                    .and_then(|a| a.last())
                    .and_then(|v| v.as_str())
                    .unwrap_or("Allow");
                let expected = if expected_str == "Allow" {
                    IdentityEffect::Allow
                } else {
                    IdentityEffect::Deny
                };

                let got = replay_scenario(&sc, defense)?;

                total += 1;
                if got == expected {
                    correct += 1;
                }

                if attack == "legit" {
                    legit_total += 1;
                    if got == IdentityEffect::Allow {
                        legit_allowed += 1;
                    } else {
                        // legit got Deny = false block
                        false_block += 1;
                    }
                } else {
                    // non-legit attack
                    let entry = per_attack.entry(attack.clone()).or_insert((0, 0));
                    entry.1 += 1; // count
                    if got == IdentityEffect::Deny {
                        entry.0 += 1; // blocked
                    } else {
                        // attack got through = attack success
                        attack_success += 1;
                    }
                }
            }

            let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
            let hcr = if legit_total > 0 { legit_allowed as f64 / legit_total as f64 } else { 1.0 };

            let per_attack_json: serde_json::Map<String, serde_json::Value> = per_attack
                .into_iter()
                .map(|(k, (blocked, count))| {
                    let block_rate = if count > 0 { blocked as f64 / count as f64 } else { 0.0 };
                    (k, serde_json::json!({ "blocked": blocked, "count": count, "block_rate": block_rate }))
                })
                .collect();

            println!("{}", serde_json::json!({
                "defense": defense,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "attack_success": attack_success,
                "false_block": false_block,
                "hcr": hcr,
                "per_attack": per_attack_json,
            }));

            Ok(exit::ALLOW)
        }
    }
}

/// Replay a single MABench scenario through the given defense rung.
/// Returns the final IdentityEffect (fail-closed: first Deny in a hop chain ends early).
///
/// Defense levels:
///   "none"         — trust all headers, always Allow (E1 baseline: no identity layer).
///   "jwt"          — bearer model: registered + envelope sig only. No cap attenuation,
///                    no provenance enforcement, no accountant. T1 passes (no ceiling
///                    check); T5 forged-prov passes (sig valid, prov not checked).
///   "biscuit"      — registered + envelope sig + capability token + registry-ceiling.
///                    NO provenance enforcement, NO accountant. T5 still passes here.
///   "biscuit_prov" — as biscuit, plus provenance chain enforcement. T5 caught here.
///                    Still no fleet accountant.
///   "full"         — all checks including the fleet Accountant from fleet_budget.
///
/// Per-hop `forged_prov` field (optional JSON array of strings): when present on a
/// handoff, the envelope is sealed with EXACTLY that array as its provenance (the
/// attacker controls their own message and signs it with their real key). At
/// biscuit_prov/full the gateway compares the envelope provenance against the REAL
/// expected chain and finds a mismatch → Deny. At jwt/biscuit provenance is not
/// enforced → Allow (signature is valid because the real signer key was used).
fn replay_scenario(sc: &serde_json::Value, defense: &str) -> crate::Result<crate::identity::IdentityEffect> {
    use crate::identity::{
        AgentId, Capability, IdentityEffect,
        accountant::Accountant,
        envelope::{seal, verify_provenance, verify_signature},
        registry::{sign_registry, Registration, Registry},
        token::{authorize, mint},
    };
    use biscuit_auth::KeyPair;
    use ed25519_dalek::SigningKey;
    use std::collections::HashMap;

    // "none" baseline: always allow, no identity layer at all.
    if defense == "none" {
        return Ok(IdentityEffect::Allow);
    }

    // ── Build the fleet from the scenario ──────────────────────────────────
    let agents_arr = sc["agents"].as_array()
        .ok_or_else(|| crate::error::Error::Config("missing agents array".into()))?;

    let issuer_seed = sc["issuer_seed"].as_u64().unwrap_or(1) as u8;
    let issuer_key = SigningKey::from_bytes(&[issuer_seed; 32]);

    // agent_id -> SigningKey (for building envelopes as the signer)
    let mut agent_keys: HashMap<String, SigningKey> = HashMap::new();
    let mut registrations: Vec<Registration> = Vec::new();

    for ag in agents_arr {
        let id = ag["id"].as_str().unwrap_or("").to_string();
        let role = ag["role"].as_str().unwrap_or("unknown").to_string();
        let key_seed = ag["key_seed"].as_u64().unwrap_or(2) as u8;
        let sk = SigningKey::from_bytes(&[key_seed; 32]);
        let max_caps: Vec<Capability> = ag["max_caps"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| Capability(s.to_string()))
            .collect();

        registrations.push(Registration {
            agent_id: AgentId(id.clone()),
            role,
            pubkey_hex: hex::encode(sk.verifying_key().to_bytes()),
            max_caps,
            issuer: "replay".to_string(),
        });
        agent_keys.insert(id, sk);
    }

    let registry_file = sign_registry(registrations, &issuer_key)?;
    let registry = Registry::load_verified(&registry_file)?;

    // ── Fleet budget (only "full" uses it) ─────────────────────────────────
    let budget: HashMap<String, u32> = if defense == "full" {
        sc["fleet_budget"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n as u32)))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        HashMap::new()
    };
    let mut accountant = Accountant::new(budget);

    // ── Fresh biscuit root keypair (internally consistent per scenario) ─────
    let root_kp = KeyPair::new();
    let root_pub = root_kp.public();

    // ── Walk the handoffs (fail-closed: first Deny ends the chain) ─────────
    let handoffs = sc["handoffs"].as_array()
        .ok_or_else(|| crate::error::Error::Config("missing handoffs array".into()))?;

    // real_prior tracks the REAL accumulated provenance chain, regardless of what
    // the attacker writes into their envelope. This is the ground truth the gateway
    // compares against when provenance enforcement is active.
    let mut real_prior: Vec<String> = Vec::new();
    let mut last_effect = IdentityEffect::Allow;

    for (hop_idx, h) in handoffs.iter().enumerate() {
        let from_str = h["from"].as_str().unwrap_or("").to_string();
        let signer_str = h["signer"].as_str().unwrap_or(&from_str).to_string();
        let action_str = h["action"].as_str().unwrap_or("").to_string();
        let claimed_caps: Vec<Capability> = h["claimed_caps"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| Capability(s.to_string()))
            .collect();
        let body = h["body"].as_str().unwrap_or("").to_string();

        let from = AgentId(from_str.clone());
        let action = Capability(action_str.clone());
        let this_hop = format!("h{hop_idx}");

        // ── Envelope provenance: attacker-controlled vs. real chain ─────────
        // When forged_prov is present the attacker seals their envelope with that
        // exact array (they control their own message). The REAL expected chain is
        // always real_prior + [this_hop] — the gateway uses this for the provenance
        // check at biscuit_prov/full.
        let env_prov: Vec<String> = if let Some(fp) = h["forged_prov"].as_array() {
            fp.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            // Honest path: real chain extended by this hop.
            let mut p = real_prior.clone();
            p.push(this_hop.clone());
            p
        };

        // ── Signing key: use the signer's real key if registered, else a ghost key ──
        // For T2-style attacks the signer is unregistered; using seed 99 (not in
        // registry) means the envelope sig check (vs. the registered key of `from`)
        // will fail → Deny. For T5 the signer IS registered (valid sig, forged prov).
        let signer_signing_key: SigningKey = agent_keys.get(&signer_str).cloned()
            .unwrap_or_else(|| SigningKey::from_bytes(&[99u8; 32]));

        // Seal the envelope with the chosen provenance and signing key.
        let env = seal(&signer_signing_key, &from, &body, env_prov)?;

        // ── Per-rung check sequence ─────────────────────────────────────────
        // Each rung applies a strict superset of the previous rung's checks.
        // Fail-closed: return Deny on the first failed check.

        // CHECK: registered (jwt, biscuit, biscuit_prov, full)
        let reg = match registry.get(&from) {
            Some(r) => r,
            None => return Ok(IdentityEffect::Deny),
        };

        // CHECK: envelope_sig (jwt, biscuit, biscuit_prov, full)
        // Verify that `env` was signed by the key registered for `from`.
        // If `signer != from` (or signer is a ghost), the keys differ → Deny.
        {
            let vk = reg.verifying_key()?;
            if verify_signature(&env, &vk).is_err() {
                return Ok(IdentityEffect::Deny);
            }
        }

        if defense == "jwt" {
            // jwt: registered + envelope_sig only. No cap token, no provenance, no accountant.
            real_prior.push(this_hop);
            last_effect = IdentityEffect::Allow;
            continue;
        }

        // CHECK: capability token + registry-ceiling (biscuit, biscuit_prov, full)
        // Mint the token with the adversary-supplied claimed_caps, then authorize
        // against the agent's registered ceiling.
        // authorize() checks that the token grants the action (T3 confused-deputy), but
        // the ceiling filter in its return value is passive — it does NOT error on an
        // above-ceiling action.  The explicit gateway ceiling gate below is the T1 defeat:
        // the registry is the root of trust; no token can grant more than max_caps.
        let token_b64 = mint(&root_kp, &from, &claimed_caps)?;
        if authorize(&token_b64, root_pub, &from, &action, &reg.max_caps).is_err() {
            return Ok(IdentityEffect::Deny);
        }
        // Gateway ceiling gate: action must be within the agent's registered ceiling.
        // Catches T1 impersonation: adversary mints a token with a cap they don't hold.
        if !reg.max_caps.contains(&action) {
            return Ok(IdentityEffect::Deny);
        }

        if defense == "biscuit" {
            // biscuit: registered + envelope_sig + capability. NO provenance, NO accountant.
            // T5 (forged_prov) slips through here because we do not call verify_provenance.
            real_prior.push(this_hop);
            last_effect = IdentityEffect::Allow;
            continue;
        }

        // CHECK: provenance chain (biscuit_prov, full ONLY)
        // Compare the envelope's provenance against the REAL expected chain.
        // A forged_prov that differs from real_prior + [this_hop] → Deny.
        if verify_provenance(&env, &real_prior, &this_hop).is_err() {
            return Ok(IdentityEffect::Deny);
        }

        if defense == "biscuit_prov" {
            // biscuit_prov: registered + envelope_sig + capability + provenance.
            // NO fleet accountant.
            real_prior.push(this_hop);
            last_effect = IdentityEffect::Allow;
            continue;
        }

        // CHECK: fleet budget (full ONLY)
        if accountant.charge(&action).is_err() {
            return Ok(IdentityEffect::Deny);
        }

        // CHECK: quorum gate (full ONLY) — k-of-n distinct, registered approvers.
        // Defeats T4 manufactured-quorum (Sybil duplicate approvers).
        let quorum_k = h["quorum_k"].as_u64().unwrap_or(0) as usize;
        if quorum_k > 0 {
            let approvers: Vec<AgentId> = h["approvers"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| AgentId(s.to_string()))
                .collect();
            if crate::identity::quorum::check_quorum(
                &approvers,
                quorum_k,
                |a| registry.get(a).is_some(),
            ).is_err() {
                return Ok(IdentityEffect::Deny);
            }
        }

        // All checks passed for this hop at "full".
        real_prior.push(this_hop);
        last_effect = IdentityEffect::Allow;
    }

    Ok(last_effect)
}

fn report(args: &ReportArgs, json: bool) -> crate::Result<i32> {
    // Peek the first non-empty line: if it parses as a chained entry (has
    // this_hash + prev_hash), warn the user that report reads plain logs only.
    if let Ok(text) = std::fs::read_to_string(&args.path) {
        if let Some(first_line) = text.lines().find(|l| !l.trim().is_empty()) {
            if crate::audit::chain::parse_line(first_line).is_ok() {
                eprintln!(
                    "warning: {} is a tamper-evident chained log; `qfire report` reads plain \
                     logs only. Use `qfire audit verify --log {}` to check integrity.",
                    args.path.display(),
                    args.path.display()
                );
            }
        }
    }
    let records = crate::audit::AuditLog::read_all(&args.path)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&records)?);
        return Ok(exit::ALLOW);
    }
    let total = records.len();
    let allowed = records.iter().filter(|r| r.terminal == Verdict::Allow).count();
    let blocked = records.iter().filter(|r| r.terminal == Verdict::Block).count();
    let avg_wall: f64 = if total > 0 {
        records.iter().map(|r| r.wall_clock_ms).sum::<f64>() / total as f64
    } else {
        0.0
    };
    let cost: f64 = records.iter().filter_map(|r| r.usage.as_ref()).map(|u| u.cost_usd).sum();
    println!("audit: {}", args.path.display());
    println!("  records:   {total}");
    println!("  allowed:   {allowed}");
    println!("  blocked:   {blocked}");
    println!("  avg wall:  {avg_wall:.1}ms");
    println!("  total cost: ${cost:.6}");
    Ok(exit::ALLOW)
}

// ── Paper 007: retrieval poison-defense CLI ───────────────────────────────────

#[derive(Subcommand)]
pub enum RetrievalCommand {
    /// Retrieve top-k for a query over a JSONL corpus, printing spotlighted hits.
    Query(RetrievalQueryArgs),
    /// Scan a JSONL corpus and report poison flags (instruction-in-data + anomaly).
    Scan(RetrievalScanArgs),
}

#[derive(Args)]
pub struct RetrievalQueryArgs {
    /// JSONL file of Documents (id, text, source, tier, signature).
    #[arg(long)]
    pub corpus: PathBuf,
    /// Free-text query string.
    #[arg(long)]
    pub query: String,
    /// Number of hits to return.
    #[arg(long, default_value_t = 5)]
    pub k: usize,
    /// TurboQuant bit-width (1, 2, or 4; 0 = exact float).
    #[arg(long, default_value_t = 2)]
    pub bits: u8,
}

#[derive(Args)]
pub struct RetrievalScanArgs {
    /// JSONL file of Documents.
    #[arg(long)]
    pub corpus: PathBuf,
    /// Instruction-in-data score threshold (0.0–1.0).
    #[arg(long, default_value_t = 0.5)]
    pub instr_threshold: f64,
}

// ── Paper 011: continuous performance & drift monitoring CLI ──────────────────

#[derive(Args)]
pub struct MonitorArgs {
    /// JSONL file of MonitorEvents (the read-side decision stream).
    #[arg(long)]
    pub stream: PathBuf,
    /// Drift detector: cusum | adwin | conformal | distdistance (default: adwin).
    #[arg(long)]
    pub detector: Option<String>,
}

fn run_monitor(args: MonitorArgs, json: bool) -> crate::Result<i32> {
    use crate::monitor::alert::AlertLadder;
    use crate::monitor::autonomy::{AutonomyEnvelope, AutonomyMeter};
    use crate::monitor::behavior::BehaviorBaseline;
    use crate::monitor::drift::{Adwin, Conformal, Cusum, DistDistance, DistMetric};
    use crate::monitor::{MonitorEvent, StreamDetector};

    let text = std::fs::read_to_string(&args.stream)?;
    let mut events: Vec<MonitorEvent> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        events.push(serde_json::from_str(line)?);
    }

    let dname = args.detector.as_deref().unwrap_or("adwin");
    let mut drift: Box<dyn StreamDetector> = match dname {
        "cusum" => Box::new(Cusum::new(0.05, 0.5, 100)),
        "adwin" => Box::new(Adwin::new(0.05)),
        "conformal" => Box::new(Conformal::new(100, 0.5, 0.01)),
        "distdistance" => Box::new(DistDistance::new(200, 100, 10, 0.2, DistMetric::Psi)),
        other => return Err(crate::error::Error::Config(format!("unknown detector '{other}'"))),
    };

    let mut autonomy = AutonomyMeter::new(AutonomyEnvelope {
        max_autonomous_risk_tier: 2,
        max_autonomous_fraction: 0.5,
        window: 100,
    });
    let mut behavior = BehaviorBaseline::new(200, 100, 0.5);
    let mut ladder = AlertLadder::default();
    let mut alerts = Vec::new();

    for ev in &events {
        if let Some(sig) = drift.observe(ev.case, ev.score) {
            alerts.push(ladder.classify(&sig));
        }
        if let Some(sig) = autonomy.observe(ev.case, ev.autonomous, ev.autonomy_level) {
            alerts.push(ladder.classify(&sig));
        }
        if let Some(sig) = behavior.observe(ev.case, &ev.tool) {
            alerts.push(ladder.classify(&sig));
        }
    }

    if json {
        println!(
            "{}",
            serde_json::json!({
                "events": events.len(),
                "detector": dname,
                "alerts": alerts,
                "clinician_volume": ladder.clinician_volume(),
            })
        );
    } else {
        for a in &alerts {
            println!("[{:?}] case {} {} ({})", a.level, a.case, a.source, a.detail);
        }
        println!(
            "— {} events, {} alerts ({} clinician-facing) via {}",
            events.len(),
            alerts.len(),
            ladder.clinician_volume(),
            dname
        );
    }
    Ok(exit::ALLOW)
}

fn run_retrieval(cmd: RetrievalCommand) -> crate::Result<i32> {
    use crate::retrieval::{broker, detect, embed::HashEmbedder, store::DocStore, Document, Embedder, RetrievalCfg};
    use crate::audit::AuditSink;

    fn load_corpus(p: &std::path::Path) -> crate::Result<Vec<Document>> {
        let text = std::fs::read_to_string(p)?;
        let mut docs = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            docs.push(serde_json::from_str(line)?);
        }
        Ok(docs)
    }

    match cmd {
        RetrievalCommand::Query(a) => {
            let docs = load_corpus(&a.corpus)?;
            let e = HashEmbedder::default();
            let store = DocStore::build(docs, &e, a.bits, 42);
            let cfg = RetrievalCfg { quant_bits: a.bits, ..Default::default() };
            // CLI: no persistent chain; use a no-op plain sink
            let sink = AuditSink::disabled();
            let hits = broker::retrieve(&store, &e, &a.query, a.k, None, &cfg, &sink)?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
            Ok(exit::ALLOW)
        }
        RetrievalCommand::Scan(a) => {
            let docs = load_corpus(&a.corpus)?;
            let e = HashEmbedder::default();
            let embs: Vec<Vec<f32>> = docs.iter().map(|d| e.embed(&d.text)).collect();
            let outliers = detect::embedding_anomaly(&embs, 2.0);
            let mut flagged = Vec::new();
            for (i, d) in docs.iter().enumerate() {
                let instr = detect::instruction_in_data(&d.text, a.instr_threshold);
                if instr || outliers.contains(&i) {
                    flagged.push(serde_json::json!({
                        "id": d.id,
                        "instruction_in_data": instr,
                        "embedding_outlier": outliers.contains(&i),
                    }));
                }
            }
            println!("{}", serde_json::json!({"flagged": flagged, "total": docs.len()}));
            Ok(exit::ALLOW)
        }
    }
}

// ─── Paper 012: regulatory passport (lifecycle) ──────────────────────────────

#[derive(Subcommand)]
pub enum LifecycleCommand {
    /// Classify agent metadata (JSON) into risk + autonomy, per jurisdiction.
    Classify(LifecycleClassifyArgs),
    /// Classify across all four jurisdictions and report explained divergences.
    Harmonize(LifecycleClassifyArgs),
    /// Verify a signed passport file (signature + expiry).
    Verify(LifecycleVerifyArgs),
    /// Run the admission deployment gate over a passport + a live fingerprint.
    Gate(LifecycleGateArgs),
}

#[derive(Args)]
pub struct LifecycleClassifyArgs {
    /// Agent metadata as inline JSON, or `@path` to read from a file.
    #[arg(long)]
    pub metadata: String,
    /// One jurisdiction: fda | eu_ai_act | health_canada | mhra (classify only;
    /// omit to print all four).
    #[arg(long)]
    pub jurisdiction: Option<String>,
}

#[derive(Args)]
pub struct LifecycleVerifyArgs {
    /// Path to a signed passport JSON file.
    #[arg(long)]
    pub passport: PathBuf,
    /// Unix-seconds "now" for the expiry check (default: 0 = signature only).
    #[arg(long, default_value_t = 0)]
    pub now: i64,
}

#[derive(Args)]
pub struct LifecycleGateArgs {
    /// Path to a signed passport JSON file.
    #[arg(long)]
    pub passport: PathBuf,
    /// Live fingerprint as inline JSON ({weights,prompt,tools,data} sha256 hex), or `@path`.
    #[arg(long)]
    pub live_fingerprint: String,
    /// Comma-separated registered agent_ids (the verified 008 registry membership).
    #[arg(long, default_value = "")]
    pub registry: String,
    /// Unix-seconds "now" for the expiry check.
    #[arg(long, default_value_t = 0)]
    pub now: i64,
    /// Enforcement: block | warn | off (default: block).
    #[arg(long, default_value = "block")]
    pub enforce: String,
}

/// Read an inline JSON argument, or load it from a file when prefixed with `@`.
fn json_arg(s: &str) -> crate::Result<String> {
    if let Some(path) = s.strip_prefix('@') {
        Ok(std::fs::read_to_string(path)?)
    } else {
        Ok(s.to_string())
    }
}

fn parse_jurisdiction(s: &str) -> crate::Result<crate::lifecycle::Jurisdiction> {
    serde_json::from_value(serde_json::Value::String(s.to_string()))
        .map_err(|_| crate::error::Error::Config(format!("unknown jurisdiction '{s}'")))
}

fn parse_enforce(s: &str) -> crate::Result<crate::lifecycle::Enforce> {
    match s {
        "block" => Ok(crate::lifecycle::Enforce::Block),
        "warn" => Ok(crate::lifecycle::Enforce::Warn),
        "off" => Ok(crate::lifecycle::Enforce::Off),
        other => Err(crate::error::Error::Config(format!("unknown enforce '{other}'"))),
    }
}

fn run_lifecycle(cmd: LifecycleCommand, json: bool) -> crate::Result<i32> {
    use crate::lifecycle::{classifier, fingerprint::Fingerprint, gate, harmonize, passport::SignedPassport, AgentMetadata, Jurisdiction};

    match cmd {
        LifecycleCommand::Classify(a) => {
            let meta: AgentMetadata = serde_json::from_str(&json_arg(&a.metadata)?)?;
            let js: Vec<Jurisdiction> = match &a.jurisdiction {
                Some(s) => vec![parse_jurisdiction(s)?],
                None => Jurisdiction::ALL.to_vec(),
            };
            let out: Vec<_> = js
                .iter()
                .map(|&j| serde_json::json!({"jurisdiction": j, "classification": classifier::classify(&meta, j)}))
                .collect();
            if json {
                println!("{}", serde_json::json!({"classifications": out}));
            } else {
                for (j, o) in js.iter().zip(&out) {
                    let c = &o["classification"];
                    println!("{:>14}  risk={:<9} autonomy={}", j.name(), c["risk_class"].as_str().unwrap_or("?"), c["autonomy_level"].as_str().unwrap_or("?"));
                }
            }
            Ok(exit::ALLOW)
        }
        LifecycleCommand::Harmonize(a) => {
            let meta: AgentMetadata = serde_json::from_str(&json_arg(&a.metadata)?)?;
            let report = harmonize::harmonize(&meta);
            if json {
                println!("{}", serde_json::to_string(&report)?);
            } else {
                println!("agreement: {:.3} ({} divergence(s))", report.agreement, report.divergences.len());
                for d in &report.divergences {
                    println!("  {} vs {}: {}", d.a.name(), d.b.name(), d.rationale);
                }
            }
            Ok(if report.divergences.is_empty() { exit::ALLOW } else { exit::ALLOW })
        }
        LifecycleCommand::Verify(a) => {
            let signed: SignedPassport = serde_json::from_str(&std::fs::read_to_string(&a.passport)?)?;
            let res = if a.now > 0 { signed.verify_at(a.now) } else { signed.verify() };
            match res {
                Ok(p) => {
                    if json {
                        println!("{}", serde_json::json!({"valid": true, "agent_id": p.agent_id, "version": p.version}));
                    } else {
                        println!("valid passport: agent_id={} version={}", p.agent_id, p.version);
                    }
                    Ok(exit::ALLOW)
                }
                Err(e) => {
                    if json {
                        println!("{}", serde_json::json!({"valid": false, "error": e.to_string()}));
                    } else {
                        println!("INVALID: {e}");
                    }
                    Ok(exit::BLOCK)
                }
            }
        }
        LifecycleCommand::Gate(a) => {
            let signed: SignedPassport = serde_json::from_str(&std::fs::read_to_string(&a.passport)?)?;
            let live: Fingerprint = serde_json::from_str(&json_arg(&a.live_fingerprint)?)?;
            let registry: Vec<String> = a.registry.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            let enforce = parse_enforce(&a.enforce)?;
            let report = gate::decide(&signed, &live, &registry, a.now, enforce);
            if json {
                println!("{}", serde_json::to_string(&report)?);
            } else {
                println!("admitted={} agent_id={} version={}", report.admitted, report.agent_id, report.passport_version);
                for r in &report.reasons {
                    println!("  - {r}");
                }
            }
            Ok(if report.admitted { exit::ALLOW } else { exit::BLOCK })
        }
    }
}
