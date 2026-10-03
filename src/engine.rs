//! The parallel evaluation engine.
//!
//! The engine evaluates a [`Chain`] against an [`LlmRequest`]: it fans out the
//! chain's referenced rules concurrently, evaluates each rule's detector
//! pipeline (sequential with short-circuit, or parallel-and-aggregate), collapses
//! per-rule verdicts into one terminal decision, and emits a complete, replayable
//! [`EvalTrace`]. A verdict cache keyed by prompt hash + node version lets
//! repeated or benchmark-replayed prompts skip recomputation.

use crate::chain::{Chain, ChainMode, FailPolicy};
use crate::detector::{Detector, DetectCtx};
use crate::ir::LlmRequest;
use crate::provider::ProviderRegistry;
use crate::rule::{CompiledRule, ShortCircuit};
use crate::verdict::{NodeVerdict, Verdict};
use crate::Result;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

/// A compiled rule bundle: the rules a chain references, ready to evaluate.
pub type CompiledRules = HashMap<String, CompiledRule>;

/// The trace of a single rule's evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleTrace {
    pub rule_id: String,
    pub rule_version: String,
    pub verdict: Verdict,
    pub nodes: Vec<NodeVerdict>,
    /// Index into `nodes` of the node that decided this rule, if any.
    pub decisive_node: Option<usize>,
    /// True if a detector error contributed under fail-closed policy.
    pub errored: bool,
}

/// The complete trace of a chain evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalTrace {
    pub chain_id: String,
    pub chain_version: String,
    pub mode: ChainMode,
    pub fail_policy: FailPolicy,
    pub rules: Vec<RuleTrace>,
    /// Wall-clock time for the whole chain (parallel fan-out).
    pub wall_clock_ms: f64,
    /// Sum of every detector node's latency (serial-equivalent work).
    pub summed_detector_ms: f64,
}

/// The terminal decision for a chain evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub terminal: Verdict,
    pub deciding_rule: Option<String>,
    pub deciding_node: Option<String>,
    pub reason: String,
    pub prompt_hash: String,
    pub trace: EvalTrace,
}

impl Decision {
    pub fn allowed(&self) -> bool {
        self.terminal == Verdict::Allow
    }
}

/// A simple verdict cache keyed by prompt hash + node identity.
#[derive(Default)]
pub struct VerdictCache {
    map: Mutex<HashMap<String, NodeVerdict>>,
}

impl VerdictCache {
    pub fn get(&self, key: &str) -> Option<NodeVerdict> {
        self.map.lock().unwrap().get(key).cloned()
    }
    pub fn put(&self, key: String, v: NodeVerdict) {
        self.map.lock().unwrap().insert(key, v);
    }
    pub fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }
}

/// The evaluation engine.
pub struct Engine {
    providers: Arc<ProviderRegistry>,
    cache: Arc<VerdictCache>,
    cache_enabled: bool,
    sem: Arc<Semaphore>,
    harness: Option<std::sync::Arc<dyn crate::harness::engine_hook::HarnessRewriter>>,
}

impl Engine {
    pub fn new(providers: Arc<ProviderRegistry>) -> Self {
        Engine {
            providers,
            cache: Arc::new(VerdictCache::default()),
            cache_enabled: true,
            sem: Arc::new(Semaphore::new(16)),
            harness: None,
        }
    }

    /// Set the maximum number of concurrently-running detector nodes.
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.sem = Arc::new(Semaphore::new(n.max(1)));
        self
    }

    /// Enable or disable the verdict cache.
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Install an adversarial/benign harness that may rewrite a rule's scope
    /// (MIIM) per request. Used to demonstrate goal tilt inside the live engine.
    pub fn with_harness(
        mut self,
        harness: std::sync::Arc<dyn crate::harness::engine_hook::HarnessRewriter>,
    ) -> Self {
        self.harness = Some(harness);
        self
    }

    pub fn providers(&self) -> &Arc<ProviderRegistry> {
        &self.providers
    }

    pub fn cache(&self) -> &Arc<VerdictCache> {
        &self.cache
    }

    /// Hash a prompt for cache keys and audit records.
    pub fn prompt_hash(prompt: &str) -> String {
        let mut h = Sha256::new();
        h.update(prompt.as_bytes());
        hex::encode(h.finalize())
    }

    /// Evaluate a chain against a request, returning a terminal decision and a
    /// full trace. `rules` must contain every rule the chain references.
    pub async fn evaluate(
        &self,
        chain: &Chain,
        rules: &HashMap<String, CompiledRule>,
        request: &LlmRequest,
    ) -> Result<Decision> {
        self.evaluate_inner(chain, rules, request, None).await
    }

    /// Post-response pass: evaluate the chain with the downstream response text
    /// attached, so response-aware detectors (e.g. output_monitor) can fire.
    pub async fn evaluate_output(
        &self,
        chain: &Chain,
        rules: &HashMap<String, CompiledRule>,
        request: &LlmRequest,
        response: &str,
    ) -> Result<Decision> {
        self.evaluate_inner(chain, rules, request, Some(response)).await
    }

    async fn evaluate_inner(
        &self,
        chain: &Chain,
        rules: &HashMap<String, CompiledRule>,
        request: &LlmRequest,
        response: Option<&str>,
    ) -> Result<Decision> {
        let prompt = request.prompt_text();
        // Audit/cache identity is over the ORIGINAL prompt; detectors may see a
        // de-obfuscated expansion when the chain enables normalization.
        let prompt_hash = Self::prompt_hash(&prompt);
        let mode = chain
            .normalize_mode
            .as_deref()
            .unwrap_or(if chain.normalize { "always" } else { "off" });
        let eval_prompt = match mode {
            "always" => crate::normalize::normalize(&prompt).expanded,
            "triggered" => {
                if crate::normalize::has_encoding_signal(&prompt) {
                    crate::normalize::normalize(&prompt).expanded
                } else {
                    prompt.clone()
                }
            }
            _ => prompt.clone(),
        };
        let referenced = chain.referenced_rules()?;

        // Validate all rule references before starting any async work.
        for rule_id in &referenced {
            if !rules.contains_key(rule_id) {
                return Err(crate::Error::Reference(format!(
                    "chain references unknown rule '{rule_id}'"
                )));
            }
        }

        // Fan out: compute effective scopes concurrently (harness rewrites in parallel).
        // Each future fetches the CompiledRule once and calls rewrite_scope; no sequential
        // loop, no double lookup.
        let effective_scopes: Vec<Option<String>> = join_all(
            referenced.iter().map(|rule_id| {
                let cr = rules.get(rule_id);
                async move {
                    match (cr, &self.harness) {
                        (Some(cr), Some(h)) => h.rewrite_scope(&cr.rule.scope, request).await,
                        _ => None,
                    }
                }
            })
        ).await;

        // wall_start measures detector evaluation only — harness rewrite latency excluded.
        let wall_start = std::time::Instant::now();

        // Build evaluate_rule futures, reusing the same CompiledRule refs from above.
        let mut futs = Vec::with_capacity(referenced.len());
        for (rule_id, effective) in referenced.iter().zip(effective_scopes.iter()) {
            let cr = rules.get(rule_id).ok_or_else(|| {
                crate::Error::Reference(format!("chain references unknown rule '{rule_id}'"))
            })?;
            futs.push(self.evaluate_rule(cr, &eval_prompt, chain.fail_policy, effective.as_deref(), response));
        }
        let traces: Vec<RuleTrace> = join_all(futs).await;
        let wall_clock_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        let by_id: HashMap<String, &RuleTrace> =
            traces.iter().map(|t| (t.rule_id.clone(), t)).collect();
        let summed_detector_ms: f64 = traces
            .iter()
            .flat_map(|t| t.nodes.iter())
            .map(|n| n.latency_ms)
            .sum();

        // Collapse to a terminal decision.
        let (terminal, deciding_rule, reason) = match chain.mode {
            ChainMode::Ordered => collapse_ordered(chain, &by_id),
            ChainMode::Expression => collapse_expression(chain, &by_id)?,
        };

        let deciding_node = deciding_rule
            .as_ref()
            .and_then(|rid| by_id.get(rid))
            .and_then(|t| t.decisive_node.map(|i| t.nodes[i].kind.clone()));

        // Order rule traces for display: ordered chains by priority, expression
        // chains by referenced order.
        let ordered_traces = order_traces(chain, traces, &referenced);

        Ok(Decision {
            terminal,
            deciding_rule,
            deciding_node,
            reason,
            prompt_hash,
            trace: EvalTrace {
                chain_id: chain.id.clone(),
                chain_version: chain.version.clone(),
                mode: chain.mode,
                fail_policy: chain.fail_policy,
                rules: ordered_traces,
                wall_clock_ms,
                summed_detector_ms,
            },
        })
    }

    /// Evaluate one rule's pipeline, honoring its short-circuit policy.
    async fn evaluate_rule(
        &self,
        cr: &CompiledRule,
        prompt: &str,
        fail: FailPolicy,
        effective_scope: Option<&str>,
        response: Option<&str>,
    ) -> RuleTrace {
        let mut ctx = DetectCtx::new(prompt, &cr.rule.scope, &cr.rule.exemplars, &self.providers);
        if let Some(eff) = effective_scope {
            ctx = ctx.with_effective_scope(eff);
        }
        if let Some(resp) = response {
            ctx = ctx.with_response(resp);
        }

        match cr.rule.short_circuit {
            ShortCircuit::Aggregate => self.collapse_aggregate(cr, &ctx, fail).await,
            ShortCircuit::StopOnFirstBlock => {
                self.collapse_stop_on_block(cr, &ctx, fail).await
            }
            ShortCircuit::StopOnFirstAllow => {
                self.collapse_stop_on_allow(cr, &ctx, fail).await
            }
        }
    }

    /// Evaluate a single node, consulting and populating the cache.
    async fn eval_node(&self, detector: &dyn Detector, prompt_hash: &str, ctx: &DetectCtx<'_>) -> NodeVerdict {
        // Include effective_scope in the key so a harness-tilted evaluation never
        // collides with a non-tilted one.  When effective_scope is None the segment
        // is empty, keeping the key stable for non-harness runs (v1 compat).
        // Include response so a pre-forward ABSTAIN/ALLOW is never served stale to a
        // post-response pass with the same prompt+effective_scope.  Empty segment
        // when response is None keeps the pre-forward key unchanged.
        let ver = detector.version();
        let key = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            for part in [
                prompt_hash,
                detector.kind(),
                ver.as_str(),
                detector.config_hash(),
                ctx.effective_scope.unwrap_or(""),
                ctx.response.unwrap_or(""),
            ] {
                h.update((part.len() as u64).to_le_bytes());
                h.update(part.as_bytes());
            }
            hex::encode(h.finalize())
        };
        if self.cache_enabled {
            if let Some(mut hit) = self.cache.get(&key) {
                hit.latency_ms = 0.0;
                if !hit.rationale.ends_with("[cache]") {
                    hit.rationale.push_str(" [cache]");
                }
                return hit;
            }
        }
        let _permit = self.sem.acquire().await;
        let v = detector.evaluate(ctx).await;
        if self.cache_enabled {
            self.cache.put(key, v.clone());
        }
        v
    }

    async fn collapse_stop_on_block(
        &self,
        cr: &CompiledRule,
        ctx: &DetectCtx<'_>,
        fail: FailPolicy,
    ) -> RuleTrace {
        let ph = Engine::prompt_hash(ctx.prompt);
        let mut nodes = Vec::new();
        let mut first_allow: Option<usize> = None;
        let mut errored = false;
        for detector in &cr.detectors {
            let v = self.eval_node(detector.as_ref(), &ph, ctx).await;
            let verdict = v.verdict;
            nodes.push(v);
            let idx = nodes.len() - 1;
            match verdict {
                Verdict::Block => {
                    return RuleTrace {
                        rule_id: cr.id().to_string(),
                        rule_version: cr.rule.version.clone(),
                        verdict: Verdict::Block,
                        nodes,
                        decisive_node: Some(idx),
                        errored,
                    };
                }
                Verdict::Error => {
                    errored = true;
                    if fail == FailPolicy::FailClosed {
                        return RuleTrace {
                            rule_id: cr.id().to_string(),
                            rule_version: cr.rule.version.clone(),
                            verdict: Verdict::Block,
                            nodes,
                            decisive_node: Some(idx),
                            errored,
                        };
                    }
                    // fail-open: treat as abstain, continue
                }
                Verdict::Allow => {
                    if first_allow.is_none() {
                        first_allow = Some(idx);
                    }
                }
                Verdict::Abstain => {}
            }
        }
        let verdict = if first_allow.is_some() { Verdict::Allow } else { Verdict::Abstain };
        RuleTrace {
            rule_id: cr.id().to_string(),
            rule_version: cr.rule.version.clone(),
            verdict,
            nodes,
            decisive_node: first_allow,
            errored,
        }
    }

    async fn collapse_stop_on_allow(
        &self,
        cr: &CompiledRule,
        ctx: &DetectCtx<'_>,
        fail: FailPolicy,
    ) -> RuleTrace {
        let ph = Engine::prompt_hash(ctx.prompt);
        let mut nodes = Vec::new();
        let mut first_block: Option<usize> = None;
        let mut errored = false;
        for detector in &cr.detectors {
            let v = self.eval_node(detector.as_ref(), &ph, ctx).await;
            let verdict = v.verdict;
            nodes.push(v);
            let idx = nodes.len() - 1;
            match verdict {
                Verdict::Allow => {
                    return RuleTrace {
                        rule_id: cr.id().to_string(),
                        rule_version: cr.rule.version.clone(),
                        verdict: Verdict::Allow,
                        nodes,
                        decisive_node: Some(idx),
                        errored,
                    };
                }
                Verdict::Block => {
                    if first_block.is_none() {
                        first_block = Some(idx);
                    }
                }
                Verdict::Error => {
                    errored = true;
                    if fail == FailPolicy::FailClosed && first_block.is_none() {
                        first_block = Some(idx);
                    }
                }
                Verdict::Abstain => {}
            }
        }
        let verdict = if first_block.is_some() { Verdict::Block } else { Verdict::Abstain };
        RuleTrace {
            rule_id: cr.id().to_string(),
            rule_version: cr.rule.version.clone(),
            verdict,
            nodes,
            decisive_node: first_block,
            errored,
        }
    }

    async fn collapse_aggregate(
        &self,
        cr: &CompiledRule,
        ctx: &DetectCtx<'_>,
        fail: FailPolicy,
    ) -> RuleTrace {
        let ph = Engine::prompt_hash(ctx.prompt);
        // Run all nodes concurrently.
        let futs = cr
            .detectors
            .iter()
            .map(|d| self.eval_node(d.as_ref(), &ph, ctx));
        let nodes: Vec<NodeVerdict> = futures::future::join_all(futs).await;

        let mut allow_w = 0.0;
        let mut block_w = 0.0;
        let mut best_block: Option<(usize, f64)> = None;
        let mut best_allow: Option<(usize, f64)> = None;
        let mut errored = false;
        for (i, n) in nodes.iter().enumerate() {
            match n.verdict {
                Verdict::Allow => {
                    allow_w += n.confidence;
                    if best_allow.map_or(true, |(_, c)| n.confidence > c) {
                        best_allow = Some((i, n.confidence));
                    }
                }
                Verdict::Block => {
                    block_w += n.confidence;
                    if best_block.map_or(true, |(_, c)| n.confidence > c) {
                        best_block = Some((i, n.confidence));
                    }
                }
                Verdict::Error => {
                    errored = true;
                    if fail == FailPolicy::FailClosed {
                        block_w += 1.0;
                        if best_block.map_or(true, |(_, c)| 1.0 > c) {
                            best_block = Some((i, 1.0));
                        }
                    }
                }
                Verdict::Abstain => {}
            }
        }
        let (verdict, decisive) = if block_w == 0.0 && allow_w == 0.0 {
            (Verdict::Abstain, None)
        } else if block_w >= allow_w {
            (Verdict::Block, best_block.map(|(i, _)| i))
        } else {
            (Verdict::Allow, best_allow.map(|(i, _)| i))
        };
        RuleTrace {
            rule_id: cr.id().to_string(),
            rule_version: cr.rule.version.clone(),
            verdict,
            nodes,
            decisive_node: decisive,
            errored,
        }
    }
}

/// Collapse an ordered chain: first BLOCK blocks, first ALLOW passes, else default.
fn collapse_ordered(
    chain: &Chain,
    by_id: &HashMap<String, &RuleTrace>,
) -> (Verdict, Option<String>, String) {
    for rule_id in &chain.rules {
        if let Some(t) = by_id.get(rule_id) {
            match t.verdict {
                Verdict::Block => {
                    return (
                        Verdict::Block,
                        Some(rule_id.clone()),
                        format!("rule '{rule_id}' blocked (ordered, first BLOCK)"),
                    );
                }
                Verdict::Allow => {
                    return (
                        Verdict::Allow,
                        Some(rule_id.clone()),
                        format!("rule '{rule_id}' allowed (ordered, first ALLOW)"),
                    );
                }
                _ => {}
            }
        }
    }
    let reason = format!("no rule decisive; applied chain default ({})", chain.default.label());
    (chain.default, None, reason)
}

/// Collapse an expression chain: the boolean DAG over "rule ALLOWed" predicates.
fn collapse_expression(
    chain: &Chain,
    by_id: &HashMap<String, &RuleTrace>,
) -> Result<(Verdict, Option<String>, String)> {
    let expr = chain.parse_expression()?;
    // A rule's predicate is "did NOT block": a guard that ABSTAINs on a clean
    // prompt passes, a scope rule that ALLOWs passes, and only an explicit BLOCK
    // makes the predicate false. This composes guard and scope rules uniformly.
    let vals: HashMap<String, bool> = by_id
        .iter()
        .map(|(k, t)| (k.clone(), t.verdict != Verdict::Block))
        .collect();
    let passed = expr.eval(&vals);
    if passed {
        Ok((Verdict::Allow, None, "expression satisfied".into()))
    } else {
        // Find a referenced rule that BLOCKED, to explain the BLOCK.
        let mut deciding = None;
        let mut names: Vec<&String> = by_id.keys().collect();
        names.sort();
        for k in names {
            if by_id.get(k).map(|t| t.verdict) == Some(Verdict::Block) {
                deciding = Some(k.clone());
                break;
            }
        }
        let reason = match &deciding {
            Some(r) => format!("expression unsatisfied; '{r}' did not pass"),
            None => "expression unsatisfied".into(),
        };
        Ok((Verdict::Block, deciding, reason))
    }
}

/// Order rule traces for display.
fn order_traces(chain: &Chain, traces: Vec<RuleTrace>, referenced: &[String]) -> Vec<RuleTrace> {
    let mut map: HashMap<String, RuleTrace> =
        traces.into_iter().map(|t| (t.rule_id.clone(), t)).collect();
    let order: Vec<String> = match chain.mode {
        ChainMode::Ordered => chain.rules.clone(),
        ChainMode::Expression => referenced.to_vec(),
    };
    let mut out = Vec::new();
    for id in order {
        if let Some(t) = map.remove(&id) {
            out.push(t);
        }
    }
    // Append any stragglers (deduped chain.rules etc.)
    let mut rest: Vec<RuleTrace> = map.into_values().collect();
    rest.sort_by(|a, b| a.rule_id.cmp(&b.rule_id));
    out.extend(rest);
    out
}
