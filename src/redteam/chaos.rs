//! Chaos / fault-injection resilience harness (E4): inject latency, dependency
//! loss, or partition at the gateway boundary and verify resilience invariants
//! hold (C3.1).
//!
//! The gateway-under-fault is modeled by two properties: whether it fails *open*
//! (allows what it would otherwise block — unsafe) or *closed*, and how much it
//! degrades availability under each fault kind.

use crate::redteam::{FaultKind, ResilienceInvariant};
use serde::Serialize;

/// A modeled gateway's behavior under fault.
#[derive(Debug, Clone, Copy)]
pub struct GatewayModel {
    /// If true, under fault the gateway allows actions it would normally block.
    pub fail_open: bool,
    /// Maximum tolerated availability degradation in [0,1] (the declared envelope).
    pub max_degradation: f64,
}

impl GatewayModel {
    /// The fail-safe-closed gateway: never allows under fault, sheds load gracefully.
    pub fn failsafe(max_degradation: f64) -> Self {
        GatewayModel { fail_open: false, max_degradation }
    }
}

/// Modeled availability degradation a fault induces before mitigation.
fn fault_degradation(kind: FaultKind, magnitude: f64) -> f64 {
    let base = match kind {
        FaultKind::Latency => 0.2,
        FaultKind::DependencyLoss => 0.5,
        FaultKind::Partition => 0.7,
    };
    (base * magnitude).clamp(0.0, 1.0)
}

/// Result of injecting one fault and checking one invariant.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FaultResult {
    pub fault: FaultKind,
    pub invariant: ResilienceInvariant,
    pub held: bool,
    pub magnitude: f64,
    pub observed_degradation: f64,
    pub detail: String,
}

/// Inject one fault and check the invariant against the modeled gateway.
pub fn inject(
    model: &GatewayModel,
    fault: FaultKind,
    invariant: ResilienceInvariant,
    magnitude: f64,
) -> FaultResult {
    let observed = fault_degradation(fault, magnitude);
    let (held, detail) = match invariant {
        ResilienceInvariant::NoUnsafeAction => {
            // A fail-open gateway under fault allows what it would block → violation.
            let held = !model.fail_open;
            let detail = if held {
                "fail-safe-closed: no unsafe action under fault".into()
            } else {
                "fail-open: gateway allowed a normally-blocked action under fault".into()
            };
            (held, detail)
        }
        ResilienceInvariant::BoundedDegradation => {
            let held = observed <= model.max_degradation;
            let detail = format!(
                "observed degradation {observed:.2} vs envelope {:.2}",
                model.max_degradation
            );
            (held, detail)
        }
    };
    FaultResult {
        fault,
        invariant,
        held,
        magnitude,
        observed_degradation: observed,
        detail,
    }
}

/// (held, total) over a batch of fault results.
pub fn invariants_held(results: &[FaultResult]) -> (usize, usize) {
    (results.iter().filter(|r| r.held).count(), results.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failsafe_closed_holds_no_unsafe_action() {
        let m = GatewayModel::failsafe(1.0);
        let r = inject(&m, FaultKind::DependencyLoss, ResilienceInvariant::NoUnsafeAction, 1.0);
        assert!(r.held);
    }

    #[test]
    fn failopen_violates_no_unsafe_action() {
        let m = GatewayModel { fail_open: true, max_degradation: 1.0 };
        let r = inject(&m, FaultKind::Partition, ResilienceInvariant::NoUnsafeAction, 1.0);
        assert!(!r.held, "fail-open violation should be caught");
    }

    #[test]
    fn degradation_within_envelope_holds() {
        let m = GatewayModel::failsafe(0.5);
        // Latency at magnitude 1.0 → 0.2 degradation ≤ 0.5 envelope.
        let r = inject(&m, FaultKind::Latency, ResilienceInvariant::BoundedDegradation, 1.0);
        assert!(r.held);
    }

    #[test]
    fn degradation_exceeds_envelope_violates() {
        let m = GatewayModel::failsafe(0.3);
        // Partition at magnitude 1.0 → 0.7 degradation > 0.3 envelope.
        let r = inject(&m, FaultKind::Partition, ResilienceInvariant::BoundedDegradation, 1.0);
        assert!(!r.held);
    }

    #[test]
    fn invariants_held_counts() {
        let m = GatewayModel::failsafe(1.0);
        let rs = vec![
            inject(&m, FaultKind::Latency, ResilienceInvariant::NoUnsafeAction, 1.0),
            inject(&m, FaultKind::Partition, ResilienceInvariant::BoundedDegradation, 1.0),
        ];
        assert_eq!(invariants_held(&rs), (2, 2));
    }
}
