//! E6 (paper 008): IdentityLayer::check latency — one full gateway check
//! (registry lookup + ed25519 verify + biscuit authorize + accountant charge).
//! Build representative fixtures ONCE, then bench the hot path only.
//!
//! Run: `cargo bench --bench identity_check`
//! Criterion prints `time: [low mid high]` in nanoseconds; convert to µs by /1000.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ed25519_dalek::SigningKey;
use qfire::identity::{AgentId, Capability, Handoff, IdentityLayer};
use qfire::identity::accountant::Accountant;
use qfire::identity::envelope::seal;
use qfire::identity::registry::{sign_registry, Registration, Registry};
use qfire::identity::token::mint;
use biscuit_auth::KeyPair;
use std::collections::HashMap;

fn caps(v: &[&str]) -> Vec<Capability> {
    v.iter().map(|s| Capability(s.to_string())).collect()
}

/// Pre-built fixture: one registered agent + valid token + valid envelope.
struct Fixture {
    registry: Registry,
    root: KeyPair,
    token: String,
    agent_key: SigningKey,
}

fn build_fixture() -> Fixture {
    // Deterministic seeds (mirrors the check_tests::legit_handoff_allows test).
    let issuer_key = SigningKey::from_bytes(&[1u8; 32]);
    let agent_key = SigningKey::from_bytes(&[2u8; 32]);

    let entry = Registration {
        agent_id: AgentId("a1".into()),
        role: "triage".into(),
        pubkey_hex: hex::encode(agent_key.verifying_key().to_bytes()),
        max_caps: caps(&["read_phi", "document"]),
        issuer: "bench-issuer".into(),
    };
    let file = sign_registry(vec![entry], &issuer_key).expect("sign_registry");
    let registry = Registry::load_verified(&file).expect("load_verified");

    let root = KeyPair::new();
    let token = mint(&root, &AgentId("a1".into()), &caps(&["read_phi"])).expect("mint");

    Fixture { registry, root, token, agent_key }
}

fn bench_identity_check(c: &mut Criterion) {
    let fix = build_fixture();

    c.bench_function("identity_check", |b| {
        b.iter(|| {
            // Accountant with empty limits = unlimited, so charge() always Ok.
            // Re-create per iteration to keep state identical (no budget depletion).
            let mut acct = Accountant::new(HashMap::new());
            let mut layer = IdentityLayer {
                registry: &fix.registry,
                root: fix.root.public(),
                accountant: &mut acct,
            };

            // Build the envelope inside the iteration (it includes the signature
            // computation; the bench measures the full check() path including
            // verify_signature, verify_provenance, token::authorize, and
            // accountant::charge — the whole hot gateway path).
            let env = seal(
                &fix.agent_key,
                &AgentId("a1".into()),
                "bench-body",
                vec!["h0".into()],
            )
            .expect("seal");

            let h = Handoff {
                from: AgentId("a1".into()),
                token_b64: fix.token.clone(),
                envelope: env,
                prior_provenance: vec![],
                this_hop: "h0".into(),
                action: Capability("read_phi".into()),
                approval: None,
            };

            let result = layer.check(black_box(&h));
            black_box(result)
        });
    });
}

criterion_group!(benches, bench_identity_check);
criterion_main!(benches);
