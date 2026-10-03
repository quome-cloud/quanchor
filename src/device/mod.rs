//! Paper 014 — device-safety broker. The first *enforcing*, fail-closed layer
//! on the actuation path: device-pinned invariants + mandatory human confirm for
//! life-critical actuation (reusing 009 oversight) + a fail-safe controller that
//! holds without the LLM + a device performance monitor. Runs after identity
//! (008) and policy (005).
//!
//! `DeviceOutcome` is a new vocabulary; the detector `Verdict` and the 009
//! `OversightOutcome` are untouched. The broker is fail-closed: any invariant
//! error, unknown parameter, or denied/timed-out confirmation ⇒ `Deny`.

pub mod confirm;
pub mod failsafe;
pub mod invariant;
pub mod monitor;
pub mod sim;

use crate::oversight::channel::{ClinicianDecision, HumanChannel};
use crate::oversight::signals::ToolRisk;
use serde::{Deserialize, Serialize};

use confirm::ConfirmGate;
use invariant::InvariantSet;

/// The three FDA-regulated device classes DeviceBench models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceClass {
    Ventilator,
    InfusionPump,
    CardiacMonitor,
}

impl DeviceClass {
    /// Stable tool name for the audit trail / escalation ticket.
    pub fn tool_name(self) -> &'static str {
        match self {
            DeviceClass::Ventilator => "ventilator",
            DeviceClass::InfusionPump => "infusion_pump",
            DeviceClass::CardiacMonitor => "cardiac_monitor",
        }
    }
}

/// IEC 60601 / FDA-style risk class; drives the mandatory-confirm gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    LifeSustaining,
    LifeSupporting,
    Monitoring,
}

impl RiskClass {
    pub fn tool_risk(self) -> ToolRisk {
        match self {
            RiskClass::LifeSustaining | RiskClass::LifeSupporting => ToolRisk::LifeCritical,
            RiskClass::Monitoring => ToolRisk::Medium,
        }
    }
    /// Life-critical actuation is *always* gated on a human (C8.3.4) — this is
    /// mandatory, not risk-conditional.
    pub fn requires_confirm(self) -> bool {
        matches!(self, RiskClass::LifeSustaining | RiskClass::LifeSupporting)
    }
}

/// A single proposed device actuation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCommand {
    pub device: DeviceClass,
    pub param: String,
    /// Requested value; boolean params (e.g. `alarm_enabled`) use 0.0/1.0.
    pub value: f64,
    #[serde(default)]
    pub patient_kg: Option<f64>,
}

/// Why an invariant rejected a command (also carries the confirm-gate refusal).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvariantViolation {
    pub param: String,
    pub kind: String,
    pub detail: String,
}

/// The broker's terminal decision for one command.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DeviceOutcome {
    Allow,
    Clamp { to: f64 },
    Deny { violation: InvariantViolation },
}

/// The device-safety broker: invariants → (life-critical) mandatory confirm.
/// Fail-closed throughout.
pub struct DeviceGuard {
    inv: InvariantSet,
    gate: ConfirmGate,
}

impl DeviceGuard {
    pub fn new(inv: InvariantSet, gate: ConfirmGate) -> Self {
        DeviceGuard { inv, gate }
    }

    pub fn with_defaults() -> Self {
        DeviceGuard {
            inv: InvariantSet::defaults(),
            gate: ConfirmGate::with_defaults(),
        }
    }

    /// Read-only view of the invariant tables (used by the harness/sim).
    pub fn invariants(&self) -> &InvariantSet {
        &self.inv
    }

    /// Check one command. Out-of-range / interlock / alarm violations deny
    /// immediately and never touch the channel. In-range life-critical commands
    /// require a human `Allow`; anything else (Deny/Timeout) is fail-closed.
    pub async fn check(&self, cmd: &DeviceCommand, ch: &dyn HumanChannel) -> DeviceOutcome {
        if let Some(v) = self.inv.check(cmd) {
            return DeviceOutcome::Deny { violation: v };
        }
        let needs = self.inv.risk_class(cmd.device).requires_confirm() || self.inv.requires_confirm(cmd);
        if !needs {
            return DeviceOutcome::Allow;
        }
        match self
            .gate
            .confirm(&format!("dev-{}-{}", cmd.device.tool_name(), cmd.param), cmd.device.tool_name(), ch)
            .await
        {
            ClinicianDecision::Allow => DeviceOutcome::Allow,
            _ => DeviceOutcome::Deny {
                violation: InvariantViolation {
                    param: cmd.param.clone(),
                    kind: "unconfirmed_life_critical".into(),
                    detail: "mandatory human confirmation not granted".into(),
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oversight::channel::OracleChannel;

    fn vent(param: &str, value: f64) -> DeviceCommand {
        DeviceCommand {
            device: DeviceClass::Ventilator,
            param: param.into(),
            value,
            patient_kg: Some(70.0),
        }
    }

    #[test]
    fn risk_class_maps_to_tool_risk() {
        assert_eq!(RiskClass::LifeSustaining.tool_risk(), ToolRisk::LifeCritical);
        assert_eq!(RiskClass::LifeSupporting.tool_risk(), ToolRisk::LifeCritical);
        assert_eq!(RiskClass::Monitoring.tool_risk(), ToolRisk::Medium);
        assert!(RiskClass::LifeSustaining.requires_confirm());
        assert!(!RiskClass::Monitoring.requires_confirm());
    }

    #[test]
    fn device_command_round_trips_json() {
        let c = vent("tidal_volume_ml", 450.0);
        let s = serde_json::to_string(&c).unwrap();
        let back: DeviceCommand = serde_json::from_str(&s).unwrap();
        assert_eq!(back.param, "tidal_volume_ml");
        assert_eq!(back.device, DeviceClass::Ventilator);
    }

    #[tokio::test]
    async fn guard_denies_out_of_range_without_touching_channel() {
        let g = DeviceGuard::with_defaults();
        let out = g.check(&vent("tidal_volume_ml", 1500.0), &OracleChannel::deny()).await;
        assert!(matches!(out, DeviceOutcome::Deny { .. }));
    }

    #[tokio::test]
    async fn guard_escalates_in_range_life_critical() {
        let g = DeviceGuard::with_defaults();
        let allow = g.check(&vent("tidal_volume_ml", 450.0), &OracleChannel::allow()).await;
        assert_eq!(allow, DeviceOutcome::Allow);
        let deny = g.check(&vent("tidal_volume_ml", 450.0), &OracleChannel::deny()).await;
        assert!(matches!(deny, DeviceOutcome::Deny { .. }));
    }

    #[tokio::test]
    async fn monitoring_in_range_allows_without_confirm() {
        let g = DeviceGuard::with_defaults();
        let cmd = DeviceCommand {
            device: DeviceClass::CardiacMonitor,
            param: "hr_alarm_high".into(),
            value: 120.0,
            patient_kg: None,
        };
        // Monitoring class -> no confirm; OracleChannel::deny is never consulted.
        let out = g.check(&cmd, &OracleChannel::deny()).await;
        assert_eq!(out, DeviceOutcome::Allow);
    }
}
