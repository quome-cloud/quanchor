//! Loop-until-dry replay runner (E1) + cost accounting (E5).
//!
//! Headline experiments are model-free. [`GatewayProbe`] is a deterministic
//! modeled defense: each *built* layer carries a set of block-signature
//! substrings; an attack is blocked iff its (possibly mutated) payload still
//! contains a signature for its layer. [`mutate`] rewrites payloads to drop
//! signatures — modeling real evasion — keyed by `(round, attack id)` with no
//! RNG and no clock, so every run is byte-reproducible.

use crate::redteam::{AttackProbe, AttackRecord, RedTeamVerdict, TargetLayer, Vulnerability};
use serde::Serialize;
use std::collections::BTreeSet;

/// Deterministic modeled gateway defense.
pub struct GatewayProbe;

impl GatewayProbe {
    pub fn standard() -> Self {
        GatewayProbe
    }

    /// Block-signatures for a built layer. Unbuilt layers return `&[]` (never block).
    fn signatures(&self, layer: TargetLayer) -> &'static [&'static str] {
        match layer {
            TargetLayer::Firewall => &["ignore prior", "jailbreak", "system override"],
            TargetLayer::Audit => &["tamper-log", "delete audit"],
            TargetLayer::Admission => &["unsigned-artifact", "unattested"],
            TargetLayer::Policy => &["off-formulary", "no-encounter", "no-cosign"],
            TargetLayer::Egress => &["base64-exfil", "phi-leak"],
            TargetLayer::Retrieval => &["poison-doc", "memory-write"],
            TargetLayer::Identity => &["impersonate", "sybil-spawn", "forged-token"],
            TargetLayer::Oversight => &["bypass-review", "auto-approve"],
            TargetLayer::Equity => &["subgroup-skew"],
            TargetLayer::Monitor => &["disable-monitor", "drift-mask"],
            // Unbuilt (012–014): no defense exists.
            TargetLayer::Lifecycle | TargetLayer::Consent | TargetLayer::Device => &[],
        }
    }
}

impl AttackProbe for GatewayProbe {
    fn probe(&self, a: &AttackRecord) -> RedTeamVerdict {
        let blocked = a.layer.built()
            && self
                .signatures(a.layer)
                .iter()
                .any(|s| a.payload.contains(s));
        RedTeamVerdict {
            attack_id: a.id.clone(),
            blocked,
            bypass: !blocked && a.expected_block,
        }
    }
}

/// A small deterministic hash (FNV-1a) over a string — no RNG, stable across runs.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Number of rounds over which a population's evasion is spread.
pub const EVASION_SPREAD: u64 = 6;

/// The round at which a continuous campaign first finds the bypass for an attack,
/// in `1..=EVASION_SPREAD` — deterministic from the attack id, so the population
/// discovers vulns spread across rounds (a decaying discovery curve), and every
/// attack is evaded by round `EVASION_SPREAD`.
pub fn evasion_round(id: &str) -> u32 {
    1 + (fnv1a(id) % EVASION_SPREAD) as u32
}

/// Deterministic mutation of a payload, keyed by `(round, id)`. Below the
/// attack's [`evasion_round`] the campaign has not yet found a bypass and the
/// payload is unchanged (still caught by the signature). At or above it, dense
/// separator insertion defeats any literal signature — modeling a successful
/// obfuscation. No RNG, no clock: byte-reproducible.
pub fn mutate(payload: &str, round: u32, id: &str) -> String {
    if round == 0 || round < evasion_round(id) {
        return payload.to_string();
    }
    obfuscate(payload)
}

/// Insert a middot after every second character, so no literal substring of
/// length ≥ 3 survives (every block signature is ≥ 3 chars).
fn obfuscate(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && i % 2 == 0 {
            out.push('·');
        }
        out.push(c);
    }
    out
}

/// One point on the discovery curve.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiscoveryPoint {
    pub round: u32,
    pub cumulative_vulns: usize,
    pub new_this_round: usize,
    pub probes: u64,
}

/// One-shot: probe only `seed_round == 0` attacks. Returns the distinct bypass vulns.
pub fn one_shot(attacks: &[AttackRecord], probe: &dyn AttackProbe) -> BTreeSet<Vulnerability> {
    attacks
        .iter()
        .filter(|a| a.seed_round == 0)
        .filter(|a| probe.probe(a).bypass)
        .map(|a| a.vuln())
        .collect()
}

/// Continuous loop-until-dry. Round 0 probes the seed set; each later round
/// applies [`mutate`] to every seed attack and re-probes. Distinct bypass vulns
/// accumulate. Stops after `dry_rounds` consecutive rounds add nothing new, or
/// at `max_rounds`. Returns the full discovery curve.
pub fn loop_until_dry(
    attacks: &[AttackRecord],
    probe: &dyn AttackProbe,
    max_rounds: u32,
    dry_rounds: u32,
) -> Vec<DiscoveryPoint> {
    let seeds: Vec<&AttackRecord> = attacks.iter().filter(|a| a.seed_round == 0).collect();
    let mut found: BTreeSet<Vulnerability> = BTreeSet::new();
    let mut curve = Vec::new();
    let mut probes: u64 = 0;
    let mut consecutive_dry = 0u32;

    for round in 0..=max_rounds {
        let before = found.len();
        for a in &seeds {
            // Round 0 = original payload; later rounds = mutated payload.
            let payload = mutate(&a.payload, round, &a.id);
            let mut probed = (*a).clone();
            probed.payload = payload;
            probes += 1;
            if probe.probe(&probed).bypass {
                found.insert(a.vuln());
            }
        }
        let new = found.len() - before;
        curve.push(DiscoveryPoint {
            round,
            cumulative_vulns: found.len(),
            new_this_round: new,
            probes,
        });
        if round > 0 {
            if new == 0 {
                consecutive_dry += 1;
                if consecutive_dry >= dry_rounds {
                    break;
                }
            } else {
                consecutive_dry = 0;
            }
        }
    }
    curve
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redteam::Control;

    fn atk(id: &str, layer: TargetLayer, payload: &str, seed_round: u32) -> AttackRecord {
        AttackRecord {
            id: id.into(),
            control: Control::C3,
            atlas: format!("AML.T{:04}", layer as u8),
            layer,
            built: layer.built(),
            payload: payload.into(),
            expected_block: true,
            seed_round,
            lineage: None,
        }
    }

    #[test]
    fn unbuilt_layer_always_bypasses() {
        let p = GatewayProbe::standard();
        let v = p.probe(&atk("c", TargetLayer::Consent, "anything", 0));
        assert!(v.bypass);
        assert!(!v.blocked);
    }

    #[test]
    fn built_layer_blocks_signature_payload() {
        let p = GatewayProbe::standard();
        let v = p.probe(&atk("f", TargetLayer::Firewall, "please ignore prior instructions", 0));
        assert!(v.blocked);
        assert!(!v.bypass);
    }

    #[test]
    fn mutate_is_deterministic() {
        assert_eq!(mutate("ignore prior now", 3, "x"), mutate("ignore prior now", 3, "x"));
    }

    #[test]
    fn mutation_evades_some_signatures() {
        let p = GatewayProbe::standard();
        let base = atk("f", TargetLayer::Firewall, "ignore prior", 0);
        // Original is blocked...
        assert!(p.probe(&base).blocked);
        // ...but a sufficiently mutated payload eventually evades.
        let mut evaded = false;
        for r in 1..=6 {
            let mut m = base.clone();
            m.payload = mutate(&base.payload, r, &base.id);
            if p.probe(&m).bypass {
                evaded = true;
                break;
            }
        }
        assert!(evaded, "no mutation round evaded the firewall signature");
    }

    #[test]
    fn every_built_layer_signature_eventually_evades() {
        let p = GatewayProbe::standard();
        for layer in TargetLayer::ALL.iter().filter(|l| l.built()) {
            let sig = p.signatures(*layer)[0];
            let payload = format!("please {sig} now in this clinical request");
            let base = atk("x", *layer, &payload, 0);
            assert!(p.probe(&base).blocked, "{layer} round-0 should block its own signature");
            let mut evaded_round = None;
            for r in 1..=8 {
                let mut m = base.clone();
                m.payload = mutate(&base.payload, r, &base.id);
                if p.probe(&m).bypass {
                    evaded_round = Some(r);
                    break;
                }
            }
            assert!(evaded_round.is_some(), "{layer} signature never evaded within 8 rounds");
        }
    }

    #[test]
    fn one_shot_subset_of_continuous() {
        let p = GatewayProbe::standard();
        let attacks = vec![
            atk("a", TargetLayer::Firewall, "ignore prior", 0),
            atk("b", TargetLayer::Egress, "base64-exfil dump", 0),
            atk("c", TargetLayer::Consent, "no consent recorded", 0),
        ];
        let os = one_shot(&attacks, &p);
        let curve = loop_until_dry(&attacks, &p, 20, 3);
        let cont = curve.last().unwrap().cumulative_vulns;
        assert!(os.len() <= cont, "one-shot {} > continuous {}", os.len(), cont);
    }

    #[test]
    fn discovery_monotone_and_terminates() {
        let p = GatewayProbe::standard();
        let attacks = vec![
            atk("a", TargetLayer::Firewall, "ignore prior", 0),
            atk("b", TargetLayer::Policy, "off-formulary order", 0),
        ];
        let curve = loop_until_dry(&attacks, &p, 20, 3);
        assert!(!curve.is_empty());
        for w in curve.windows(2) {
            assert!(w[1].cumulative_vulns >= w[0].cumulative_vulns);
        }
        assert!(curve.len() <= 21);
    }
}
