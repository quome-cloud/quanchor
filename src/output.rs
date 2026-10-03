//! Human-readable and `--json` rendering of decisions and traces.
//!
//! Verdict output is dense and scannable: a single decisive line (ALLOW/BLOCK
//! with the deciding rule), then an indented per-node trace showing each
//! detector's verdict, confidence and latency, colored green for allow / red for
//! block / dim for abstain, with monospace alignment.

use crate::engine::{Decision, EvalTrace, RuleTrace};
use crate::verdict::{NodeVerdict, Verdict};
use owo_colors::OwoColorize;

/// Whether to emit ANSI colors (disabled for `--quiet`, pipes, or `NO_COLOR`).
pub fn use_color() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::IsTerminal::is_terminal(&std::io::stdout())
}

fn colorize_verdict(v: Verdict, s: &str, color: bool) -> String {
    if !color {
        return s.to_string();
    }
    match v {
        Verdict::Allow => s.green().bold().to_string(),
        Verdict::Block => s.red().bold().to_string(),
        Verdict::Abstain => s.dimmed().to_string(),
        Verdict::Error => s.yellow().bold().to_string(),
    }
}

/// Render the one-line headline for a decision.
pub fn headline(decision: &Decision, color: bool) -> String {
    let label = colorize_verdict(decision.terminal, decision.terminal.label(), color);
    let rule = decision
        .deciding_rule
        .as_deref()
        .map(|r| format!(" by {r}"))
        .unwrap_or_default();
    let node = decision
        .deciding_node
        .as_deref()
        .map(|n| format!(" [{n}]"))
        .unwrap_or_default();
    format!(
        "{label}{rule}{node} — {} ({:.1}ms wall / {:.1}ms detectors)",
        decision.reason, decision.trace.wall_clock_ms, decision.trace.summed_detector_ms
    )
}

fn render_node(n: &NodeVerdict, color: bool, indent: &str) -> String {
    let v = colorize_verdict(n.verdict, &format!("{:<8}", n.verdict.label()), color);
    let score = n.score.map(|s| format!(" score={s:.3}")).unwrap_or_default();
    format!(
        "{indent}{v} {:<10} conf={:.2} {:>6.1}ms{score}  {} ({})",
        n.kind, n.confidence, n.latency_ms, n.rationale, n.version
    )
}

fn render_rule(rt: &RuleTrace, color: bool) -> String {
    let v = colorize_verdict(rt.verdict, &format!("{:<8}", rt.verdict.label()), color);
    let mut out = format!("  {v} rule {} @{}\n", rt.rule_id, rt.rule_version);
    for (i, n) in rt.nodes.iter().enumerate() {
        let marker = if Some(i) == rt.decisive_node { "→ " } else { "  " };
        out.push_str(&render_node(n, color, &format!("    {marker}")));
        out.push('\n');
    }
    out
}

/// Render the full decision (headline + per-rule, per-node trace).
pub fn render_decision(decision: &Decision, color: bool) -> String {
    let mut out = String::new();
    out.push_str(&headline(decision, color));
    out.push('\n');
    for rt in &decision.trace.rules {
        out.push_str(&render_rule(rt, color));
    }
    out
}

/// Render an evaluation trace as a tree (used by `qfire rules explain`).
pub fn render_tree(trace: &EvalTrace, color: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "chain {} @{} [{:?} / {:?}]\n",
        trace.chain_id, trace.chain_version, trace.mode, trace.fail_policy
    ));
    let n = trace.rules.len();
    for (ri, rt) in trace.rules.iter().enumerate() {
        let branch = if ri + 1 == n { "└─" } else { "├─" };
        let rv = colorize_verdict(rt.verdict, rt.verdict.label(), color);
        out.push_str(&format!("{branch} {rv} {} @{}\n", rt.rule_id, rt.rule_version));
        let child_prefix = if ri + 1 == n { "   " } else { "│  " };
        let nn = rt.nodes.len();
        for (ni, node) in rt.nodes.iter().enumerate() {
            let nbranch = if ni + 1 == nn { "└─" } else { "├─" };
            let nv = colorize_verdict(node.verdict, node.verdict.label(), color);
            let mark = if Some(ni) == rt.decisive_node { " *" } else { "" };
            out.push_str(&format!(
                "{child_prefix}{nbranch} {nv} {} conf={:.2} {:.1}ms{mark} — {}\n",
                node.kind, node.confidence, node.latency_ms, node.rationale
            ));
        }
    }
    out
}

/// Build an OpenAI-style `chat.completion` carrying a firewall refusal as the
/// assistant message, so OpenAI-SDK clients see a normal 200 completion on BLOCK.
pub fn openai_refusal_completion(decision: &Decision, model: &str, redact: bool) -> serde_json::Value {
    use std::time::{SystemTime, UNIX_EPOCH};
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let hash = &decision.prompt_hash[..decision.prompt_hash.len().min(8)];
    const REFUSAL_MSG: &str = "I can't help with that request. It was blocked by the security policy.";
    let content = if redact {
        REFUSAL_MSG.to_string()
    } else {
        format!("{REFUSAL_MSG} (reason: {})", decision.reason)
    };
    serde_json::json!({
        "id": format!("chatcmpl-qfire-{hash}"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })
}

/// A structured refusal envelope returned by the proxy on BLOCK.
pub fn refusal_json(decision: &Decision, redact: bool) -> serde_json::Value {
    let reason = if redact { "request blocked by firewall policy".to_string() } else { decision.reason.clone() };
    serde_json::json!({
        "qfire": {
            "decision": "block",
            "blocking_rule": decision.deciding_rule,
            "blocking_node": decision.deciding_node,
            "reason": reason,
            "chain": decision.trace.chain_id,
            "prompt_hash": decision.prompt_hash,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{ChainMode, FailPolicy};
    use crate::engine::EvalTrace;

    fn block_decision() -> Decision {
        Decision {
            terminal: Verdict::Block,
            deciding_rule: Some("hipaa_phi".to_string()),
            deciding_node: Some("regex".to_string()),
            reason: "matched exfiltration pattern".to_string(),
            prompt_hash: "deadbeefcafef00d".to_string(),
            trace: EvalTrace {
                chain_id: "e7_agent".to_string(),
                chain_version: "1".to_string(),
                mode: ChainMode::Ordered,
                fail_policy: FailPolicy::FailClosed,
                rules: vec![],
                wall_clock_ms: 0.0,
                summed_detector_ms: 0.0,
            },
        }
    }

    #[test]
    fn openai_refusal_completion_shape() {
        let d = block_decision();
        let v = openai_refusal_completion(&d, "llama3.1:8b", false);
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], "llama3.1:8b");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        let content = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(!content.is_empty());
        assert!(content.contains("matched exfiltration pattern"));
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["id"], "chatcmpl-qfire-deadbeef");
    }

    #[test]
    fn openai_refusal_completion_redacts() {
        let d = block_decision();
        let v = openai_refusal_completion(&d, "gpt-4o", true);
        let content = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(!content.is_empty());
        assert!(content.contains("blocked by the security policy"));
        assert!(!content.contains("matched exfiltration pattern"));
    }
}
