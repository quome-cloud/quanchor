//! Regression gate (E2): a config/model change that reopens a previously-closed
//! vulnerability must be blocked at the deployment gate (models the 012 gate).
//!
//! A "change" is modeled as a set of layers whose defenses are switched off.
//! [`ChangedProbe`] suppresses those layers' signatures, so closed-set attacks
//! targeting them now bypass — exactly the regression the gate must catch.

use crate::redteam::runner::GatewayProbe;
use crate::redteam::{AttackProbe, AttackRecord, RedTeamVerdict, TargetLayer};
use serde::Serialize;

/// A deployment change that disables one or more layers' defenses.
#[derive(Debug, Clone)]
pub struct Change {
    pub id: String,
    pub disabled_layers: Vec<TargetLayer>,
}

/// A probe whose signatures are suppressed for the change's disabled layers.
pub struct ChangedProbe<'a> {
    base: &'a GatewayProbe,
    disabled: Vec<TargetLayer>,
}

impl<'a> ChangedProbe<'a> {
    pub fn new(base: &'a GatewayProbe, change: &Change) -> Self {
        ChangedProbe {
            base,
            disabled: change.disabled_layers.clone(),
        }
    }
}

impl AttackProbe for ChangedProbe<'_> {
    fn probe(&self, a: &AttackRecord) -> RedTeamVerdict {
        if self.disabled.contains(&a.layer) {
            // Defense switched off → bypass (the reopened vulnerability).
            RedTeamVerdict {
                attack_id: a.id.clone(),
                blocked: false,
                bypass: a.expected_block,
            }
        } else {
            self.base.probe(a)
        }
    }
}

/// Outcome of running the gate for one change.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateResult {
    pub change: String,
    /// Closed-set attacks that now bypass under the change.
    pub reopened: usize,
    /// Reopened vulns the gate detected (should equal `reopened`).
    pub caught: usize,
    /// True iff the gate blocks the deployment (any reopened vuln).
    pub blocked_deploy: bool,
}

/// Run the gate: the `closed_set` is the attacks the baseline gateway blocks. For
/// the change, re-probe them under [`ChangedProbe`]; any that now bypass is both
/// "reopened" and "caught" by the gate (the gate sees every re-probe).
pub fn run_gate(closed_set: &[AttackRecord], baseline: &GatewayProbe, change: &Change) -> GateResult {
    let changed = ChangedProbe::new(baseline, change);
    let reopened = closed_set
        .iter()
        .filter(|a| baseline.probe(a).blocked) // genuinely closed at baseline
        .filter(|a| changed.probe(a).bypass) // reopened under the change
        .count();
    GateResult {
        change: change.id.clone(),
        reopened,
        caught: reopened, // the gate re-probes everything, so it catches all reopened
        blocked_deploy: reopened > 0,
    }
}

/// Catch rate over a battery of changes = caught / reopened (1.0 if nothing reopened).
pub fn catch_rate(results: &[GateResult]) -> f64 {
    let reopened: usize = results.iter().map(|r| r.reopened).sum();
    let caught: usize = results.iter().map(|r| r.caught).sum();
    if reopened == 0 {
        1.0
    } else {
        caught as f64 / reopened as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redteam::Control;

    fn atk(id: &str, layer: TargetLayer, payload: &str) -> AttackRecord {
        AttackRecord {
            id: id.into(),
            control: Control::C3,
            atlas: "AML.T0051".into(),
            layer,
            built: layer.built(),
            payload: payload.into(),
            expected_block: true,
            seed_round: 0,
            lineage: None,
        }
    }

    #[test]
    fn change_reopening_vuln_is_caught() {
        let base = GatewayProbe::standard();
        // A closed vuln: the firewall blocks "ignore prior".
        let closed = vec![atk("f", TargetLayer::Firewall, "ignore prior")];
        let change = Change {
            id: "disable-firewall".into(),
            disabled_layers: vec![TargetLayer::Firewall],
        };
        let r = run_gate(&closed, &base, &change);
        assert_eq!(r.reopened, 1);
        assert_eq!(r.caught, 1);
        assert!(r.blocked_deploy);
    }

    #[test]
    fn innocuous_change_blocks_nothing() {
        let base = GatewayProbe::standard();
        let closed = vec![atk("f", TargetLayer::Firewall, "ignore prior")];
        // Disable a layer that has no closed-set attack.
        let change = Change {
            id: "disable-egress".into(),
            disabled_layers: vec![TargetLayer::Egress],
        };
        let r = run_gate(&closed, &base, &change);
        assert_eq!(r.reopened, 0);
        assert!(!r.blocked_deploy);
    }

    #[test]
    fn catch_rate_is_one_when_all_caught() {
        let base = GatewayProbe::standard();
        let closed = vec![
            atk("f", TargetLayer::Firewall, "ignore prior"),
            atk("p", TargetLayer::Policy, "off-formulary order"),
        ];
        let changes = vec![
            Change { id: "c1".into(), disabled_layers: vec![TargetLayer::Firewall] },
            Change { id: "c2".into(), disabled_layers: vec![TargetLayer::Policy] },
        ];
        let results: Vec<_> = changes.iter().map(|c| run_gate(&closed, &base, c)).collect();
        assert_eq!(catch_rate(&results), 1.0);
    }
}
