//! Paper-ready benchmark report artifacts: console tables, CSV, JSON, Markdown.

use super::{ChainReport, Manifest, Metrics};
use crate::Result;
use std::fmt::Write as _;
use std::path::Path;

/// Render the console summary (per-chain and per-rule tables).
pub fn render_console(manifest: &Manifest, reports: &[ChainReport]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\nQFIRE bench — seed {} — model {} — {} attacks / {} benign",
        manifest.seed, manifest.model, manifest.attack_count, manifest.benign_count
    );
    let _ = writeln!(out, "{}", "=".repeat(96));
    let _ = writeln!(
        out,
        "{:<22} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8}",
        "CHAIN / rule", "block", "inj", "fpr", "prec", "rec", "f1", "auc", "p95ms", "detms"
    );
    let _ = writeln!(out, "{}", "-".repeat(96));
    for r in reports {
        row(&mut out, &format!("{} @{}", r.chain, r.chain_version), &r.overall);
        for (rid, m) in &r.per_rule {
            row(&mut out, &format!("  {rid}"), m);
        }
        if let Some(aip) = &r.attack_in_prompt {
            row(&mut out, "  [attack-in-prompt]", aip);
        }
        // after the existing per-chain summary line(s)
        if r.throughput_qps > 0.0 {
            let _ = out.push_str(&format!(
                "  {} : {:.1} prompts/s ({:.0} ms total wall)\n",
                r.chain, r.throughput_qps, r.total_wall_ms
            ));
        }
        let _ = writeln!(out, "{}", "-".repeat(96));
    }
    out
}

fn row(out: &mut String, label: &str, m: &Metrics) {
    let _ = writeln!(
        out,
        "{:<22} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>8.3} {:>8.1} {:>8.1}",
        truncate(label, 22),
        m.block_rate,
        m.injection_rate,
        m.fpr,
        m.precision,
        m.recall,
        m.f1,
        m.auc,
        m.p95_ms,
        m.mean_detector_ms
    );
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n.saturating_sub(1)).collect::<String>() + "…"
    }
}

/// Write the full results as JSON.
pub fn write_json(dir: &Path, manifest: &Manifest, reports: &[ChainReport]) -> Result<()> {
    let val = serde_json::json!({ "manifest": manifest, "reports": reports });
    std::fs::write(dir.join("bench.json"), serde_json::to_string_pretty(&val)?)?;
    Ok(())
}

/// Write per-chain and per-rule rows as CSV.
pub fn write_csv(dir: &Path, reports: &[ChainReport]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(dir.join("bench.csv"))?;
    wtr.write_record([
        "chain", "scope", "kind", "attacks", "benign", "tp", "fp", "tn", "fn", "block_rate",
        "injection_rate", "fpr", "fnr", "precision", "recall", "f1", "accuracy", "auc", "p50_ms",
        "p95_ms", "p99_ms", "mean_wall_ms", "mean_detector_ms",
    ])
    .map_err(csv_err)?;
    for r in reports {
        write_metrics_row(&mut wtr, &r.chain, &r.chain, "chain", &r.overall)?;
        for (rid, m) in &r.per_rule {
            write_metrics_row(&mut wtr, &r.chain, rid, "rule", m)?;
        }
        if let Some(aip) = &r.attack_in_prompt {
            write_metrics_row(&mut wtr, &r.chain, "attack_in_prompt", "aip", aip)?;
        }
    }
    wtr.flush()?;
    Ok(())
}

fn write_metrics_row(
    wtr: &mut csv::Writer<std::fs::File>,
    chain: &str,
    scope: &str,
    kind: &str,
    m: &Metrics,
) -> Result<()> {
    wtr.write_record([
        chain,
        scope,
        kind,
        &m.attacks.to_string(),
        &m.benign.to_string(),
        &m.tp.to_string(),
        &m.fp.to_string(),
        &m.tn.to_string(),
        &m.fn_.to_string(),
        &fmt(m.block_rate),
        &fmt(m.injection_rate),
        &fmt(m.fpr),
        &fmt(m.fnr),
        &fmt(m.precision),
        &fmt(m.recall),
        &fmt(m.f1),
        &fmt(m.accuracy),
        &fmt(m.auc),
        &fmt(m.p50_ms),
        &fmt(m.p95_ms),
        &fmt(m.p99_ms),
        &fmt(m.mean_wall_ms),
        &fmt(m.mean_detector_ms),
    ])
    .map_err(csv_err)
}

fn fmt(x: f64) -> String {
    format!("{x:.4}")
}

fn csv_err(e: csv::Error) -> crate::Error {
    crate::Error::Other(format!("csv: {e}"))
}

/// Write a paper-ready Markdown report with a manifest header.
pub fn write_markdown(dir: &Path, manifest: &Manifest, reports: &[ChainReport]) -> Result<()> {
    let mut out = String::new();
    let _ = writeln!(out, "# QFIRE Benchmark Report\n");
    let _ = writeln!(out, "## Run manifest\n");
    let _ = writeln!(out, "| field | value |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| qfire version | {} |", manifest.qfire_version);
    let _ = writeln!(out, "| timestamp | {} |", manifest.timestamp);
    let _ = writeln!(out, "| seed | {} |", manifest.seed);
    let _ = writeln!(out, "| model | {} |", manifest.model);
    let _ = writeln!(out, "| chains | {} |", manifest.chains.join(", "));
    let _ = writeln!(out, "| attack prompts | {} |", manifest.attack_count);
    let _ = writeln!(out, "| benign prompts | {} |", manifest.benign_count);
    let _ = writeln!(out, "| attack-in-prompt | {} |\n", manifest.attack_in_prompt);

    let _ = writeln!(out, "## Per-chain results\n");
    let _ = writeln!(
        out,
        "| chain | block | inj.rate | FPR | precision | recall | F1 | AUC | p95 ms | det ms |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|");
    for r in reports {
        md_row(&mut out, &format!("`{}` @{}", r.chain, r.chain_version), &r.overall);
        if let Some(aip) = &r.attack_in_prompt {
            md_row(&mut out, &format!("`{}` (attack-in-prompt)", r.chain), aip);
        }
    }

    for r in reports {
        let _ = writeln!(out, "\n### Per-rule — `{}`\n", r.chain);
        let _ = writeln!(
            out,
            "| rule | block | inj.rate | FPR | precision | recall | F1 | AUC |"
        );
        let _ = writeln!(out, "|---|---|---|---|---|---|---|---|");
        for (rid, m) in &r.per_rule {
            let _ = writeln!(
                out,
                "| `{rid}` | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} |",
                m.block_rate, m.injection_rate, m.fpr, m.precision, m.recall, m.f1, m.auc
            );
        }
        if !r.per_task.is_empty() {
            let _ = writeln!(out, "\n### Per-task matrix\n");
            let _ = writeln!(
                out,
                "| task | attacks | block_rate | recall | f1 | tilt_score |"
            );
            let _ = writeln!(out, "|---|---|---|---|---|---|");
            for (task, m) in &r.per_task {
                let _ = writeln!(
                    out,
                    "| {task} | {} | {:.3} | {:.3} | {:.3} | {:.3} |",
                    m.attacks, m.block_rate, m.recall, m.f1, m.tilt_score
                );
            }
        }
    }

    let _ = writeln!(
        out,
        "\n_Cost note: under the local Ollama provider every LLM-judge call is $0; \
         firewall overhead is reported as latency (p95 wall-clock and summed detector time). \
         With a paid provider, cost is computed from the judge node's token usage via the \
         versioned pricing table._"
    );

    std::fs::write(dir.join("report.md"), out)?;
    Ok(())
}

fn md_row(out: &mut String, label: &str, m: &Metrics) {
    let _ = writeln!(
        out,
        "| {label} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} | {:.1} | {:.1} |",
        m.block_rate, m.injection_rate, m.fpr, m.precision, m.recall, m.f1, m.auc, m.p95_ms,
        m.mean_detector_ms
    );
}
