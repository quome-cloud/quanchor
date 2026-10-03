//! The `qfire rules` subcommands: list, lint, test, explain.

use super::{exit, PromptInput};
use crate::app::App;
use crate::output;
use crate::verdict::Verdict;
use clap::{Args, Subcommand};
use std::path::Path;

/// The global options the rules subcommands need.
#[derive(Clone, Copy)]
pub struct Globals<'a> {
    pub config: Option<&'a Path>,
    pub json: bool,
    pub quiet: bool,
}

#[derive(Subcommand)]
pub enum RulesCommand {
    /// List all rules in the library.
    List,
    /// Validate rule and chain schema + detector references.
    Lint,
    /// Run each rule against its in-scope/out-of-scope fixtures.
    Test(TestArgs),
    /// Dry-run a chain against a prompt and print the full collapse trace.
    Explain(ExplainArgs),
}

#[derive(Args)]
pub struct TestArgs {
    /// Test only this rule (default: all).
    #[arg(long)]
    pub rule: Option<String>,
}

#[derive(Args)]
pub struct ExplainArgs {
    #[command(flatten)]
    pub input: PromptInput,
    #[arg(long, short = 'c', default_value = "default")]
    pub chain: String,
}

pub async fn run(
    cmd: RulesCommand,
    config: Option<&Path>,
    json: bool,
    quiet: bool,
) -> crate::Result<i32> {
    let g = Globals { config, json, quiet };
    match cmd {
        RulesCommand::List => list(g),
        RulesCommand::Lint => lint(g),
        RulesCommand::Test(args) => test(g, args).await,
        RulesCommand::Explain(args) => explain(g, args).await,
    }
}

fn list(cli: Globals) -> crate::Result<i32> {
    let app = App::load(cli.config)?;
    if cli.json {
        let rules: Vec<_> = app.rules.all().into_iter().cloned().collect();
        println!("{}", serde_json::to_string_pretty(&rules)?);
        return Ok(exit::ALLOW);
    }
    println!("{:<28} {:<14} {:>5} {:>6}  {}", "ID", "DOMAIN", "NODES", "FIXTS", "SCOPE");
    for r in app.rules.all() {
        let domain = r.domain.clone().unwrap_or_else(|| "-".into());
        let fx = r.effective_fixtures();
        let nf = fx.in_scope.len() + fx.out_of_scope.len();
        let scope: String = r.scope.chars().take(50).collect();
        println!(
            "{:<28} {:<14} {:>5} {:>6}  {}",
            r.id,
            domain,
            r.pipeline.len(),
            nf,
            scope
        );
    }
    println!("\n{} rules, {} chains", app.rules.len(), app.chains.len());
    Ok(exit::ALLOW)
}

fn lint(cli: Globals) -> crate::Result<i32> {
    let app = App::load(cli.config)?;
    let mut problems = Vec::new();

    for r in app.rules.all() {
        if let Err(e) = r.compile() {
            problems.push(format!("rule '{}': {e}", r.id));
        }
        if r.scope.trim().is_empty() {
            problems.push(format!("rule '{}': empty scope", r.id));
        }
    }
    for (id, chain) in &app.chains {
        match chain.referenced_rules() {
            Ok(refs) => {
                for rid in refs {
                    if !app.rules.contains(&rid) {
                        problems.push(format!("chain '{id}': references unknown rule '{rid}'"));
                    }
                }
            }
            Err(e) => problems.push(format!("chain '{id}': {e}")),
        }
    }

    if cli.json {
        let val = serde_json::json!({
            "ok": problems.is_empty(),
            "rules": app.rules.len(),
            "chains": app.chains.len(),
            "problems": problems,
        });
        println!("{}", serde_json::to_string_pretty(&val)?);
    } else if problems.is_empty() {
        println!(
            "ok: {} rules, {} chains lint clean",
            app.rules.len(),
            app.chains.len()
        );
    } else {
        for p in &problems {
            eprintln!("lint: {p}");
        }
        eprintln!("{} problem(s)", problems.len());
    }
    Ok(if problems.is_empty() { exit::ALLOW } else { exit::ERROR })
}

#[derive(serde::Serialize)]
struct RuleTestResult {
    rule: String,
    passed: usize,
    failed: usize,
    failures: Vec<String>,
}

async fn test(cli: Globals<'_>, args: TestArgs) -> crate::Result<i32> {
    let app = App::load(cli.config)?.without_audit();
    let rule_ids: Vec<String> = match args.rule {
        Some(r) => vec![r],
        None => app.rules.ids(),
    };

    let mut results = Vec::new();
    let mut total_pass = 0;
    let mut total_fail = 0;

    for rid in &rule_ids {
        let rule = app.rules.get(rid)?;
        let fx = rule.effective_fixtures();
        let mut passed = 0;
        let mut failed = 0;
        let mut failures = Vec::new();

        // in_scope fixtures should NOT be blocked.
        for p in &fx.in_scope {
            let req = crate::ir::LlmRequest::user("test", p);
            let d = app.check(rid, &req).await?;
            if d.terminal == Verdict::Block {
                failed += 1;
                failures.push(format!("false-positive (blocked in-scope): {}", truncate(p)));
            } else {
                passed += 1;
            }
        }
        // out_of_scope fixtures SHOULD be blocked.
        for p in &fx.out_of_scope {
            let req = crate::ir::LlmRequest::user("test", p);
            let d = app.check(rid, &req).await?;
            if d.terminal == Verdict::Block {
                passed += 1;
            } else {
                failed += 1;
                failures.push(format!("false-negative (allowed out-of-scope): {}", truncate(p)));
            }
        }
        total_pass += passed;
        total_fail += failed;
        results.push(RuleTestResult { rule: rid.clone(), passed, failed, failures });
    }

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        for r in &results {
            let status = if r.failed == 0 { "PASS" } else { "FAIL" };
            println!("{status} {:<28} {} passed, {} failed", r.rule, r.passed, r.failed);
            if !cli.quiet {
                for f in &r.failures {
                    println!("       {f}");
                }
            }
        }
        println!("\ntotal: {total_pass} passed, {total_fail} failed");
    }
    Ok(if total_fail == 0 { exit::ALLOW } else { exit::BLOCK })
}

async fn explain(cli: Globals<'_>, args: ExplainArgs) -> crate::Result<i32> {
    let app = App::load(cli.config)?.without_audit();
    let model = super::default_model(&app);
    let req = args.input.into_request(&model)?;
    let decision = app.explain(&args.chain, &req).await?;
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&decision)?);
    } else {
        let color = !cli.quiet && output::use_color();
        println!("{}", output::headline(&decision, color));
        println!();
        print!("{}", output::render_tree(&decision.trace, color));
    }
    Ok(super::exit_for(decision.terminal))
}

fn truncate(s: &str) -> String {
    let t: String = s.chars().take(60).collect();
    if s.chars().count() > 60 {
        format!("{t}…")
    } else {
        t
    }
}
