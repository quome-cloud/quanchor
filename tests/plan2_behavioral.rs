//! Plan 2B — behavioral output monitor fires on a post-response pass.
use qfire::chain::Chain;
use qfire::config::Config;
use qfire::engine::Engine;
use qfire::ir::LlmRequest;
use qfire::provider::ProviderRegistry;
use qfire::rule::{CompiledRule, Rule};
use qfire::verdict::Verdict;
use std::collections::HashMap;
use std::sync::Arc;

fn setup() -> (Chain, HashMap<String, CompiledRule>) {
    let rule = Rule::from_yaml(
        "id: dose_output\nscope: \"dosing safety\"\nshort_circuit: stop_on_first_block\npipeline:\n  - type: output_monitor\n    safe_range: [70.0, 180.0]\n",
    ).unwrap();
    let mut rules = HashMap::new();
    rules.insert(rule.id.clone(), rule.compile().unwrap());
    let chain = Chain::from_yaml("id: t\nmode: expression\nexpression: \"dose_output\"\n").unwrap();
    (chain, rules)
}

#[tokio::test]
async fn unsafe_response_blocked_post_response() {
    let (chain, rules) = setup();
    let providers = Arc::new(ProviderRegistry::from_profiles(&Config::default().providers).unwrap());
    let engine = Engine::new(providers);
    let req = LlmRequest::user("m", "dose?");
    // Pre-forward pass: output_monitor abstains (no response) -> not a Block.
    let pre = engine.evaluate(&chain, &rules, &req).await.unwrap();
    assert_ne!(pre.terminal, Verdict::Block);
    // Post-response with unsafe value -> Block.
    let post = engine.evaluate_output(&chain, &rules, &req, "give 55 units to crash it fast").await.unwrap();
    assert_eq!(post.terminal, Verdict::Block);
    // Safe response -> not a Block.
    let ok = engine.evaluate_output(&chain, &rules, &req, "aim for ~120 in range").await.unwrap();
    assert_ne!(ok.terminal, Verdict::Block);
}
