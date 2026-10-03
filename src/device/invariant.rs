//! Per-device safe-range + interlock + alarm-preservation enforcer. The tables
//! are version-pinned and embedded from the committed
//! `datasets/014-device-broker/devicebench/devices.json`; a test locks
//! `defaults()` to that file (the 009 calibration pattern), so the shipped
//! invariants can never silently drift from the audited table.

use super::{DeviceClass, DeviceCommand, InvariantViolation, RiskClass};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeRange {
    pub min: f64,
    pub max: f64,
    #[serde(default)]
    pub requires_confirm: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interlock {
    pub left: String,
    /// Only `lt` is used today; kept as a string for forward-compat tables.
    pub op: String,
    pub right: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceSpec {
    pub risk_class: RiskClass,
    pub params: BTreeMap<String, SafeRange>,
    #[serde(default)]
    pub interlocks: Vec<Interlock>,
    #[serde(default)]
    pub alarm_params: Vec<String>,
    #[serde(default)]
    pub safe_state: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvariantSet {
    pub version: String,
    /// Keyed by `DeviceClass::tool_name()` (string keys for portable JSON).
    pub devices: BTreeMap<String, DeviceSpec>,
}

const TABLE_JSON: &str =
    include_str!("../../datasets/014-device-broker/devicebench/devices.json");

impl InvariantSet {
    /// The audited, version-pinned tables shipped with the binary.
    pub fn defaults() -> Self {
        serde_json::from_str(TABLE_JSON).expect("embedded devices.json is valid")
    }

    pub fn spec(&self, device: DeviceClass) -> Option<&DeviceSpec> {
        self.devices.get(device.tool_name())
    }

    pub fn risk_class(&self, device: DeviceClass) -> RiskClass {
        // Fail-closed: an unknown device is treated as life-sustaining.
        self.spec(device).map(|s| s.risk_class).unwrap_or(RiskClass::LifeSustaining)
    }

    /// Does this specific command's parameter demand human confirmation?
    pub fn requires_confirm(&self, cmd: &DeviceCommand) -> bool {
        self.spec(cmd.device)
            .and_then(|s| s.params.get(&cmd.param))
            .map(|r| r.requires_confirm)
            .unwrap_or(true) // unknown param -> fail closed
    }

    /// Stateless safety check for one command: range (V1), alarm-preservation
    /// (V2), and unknown-parameter / class-integrity (V3). Returns the first
    /// violation, or `None` if the command is in-policy. Fail-closed on anything
    /// unrecognised.
    pub fn check(&self, cmd: &DeviceCommand) -> Option<InvariantViolation> {
        let spec = match self.spec(cmd.device) {
            Some(s) => s,
            None => {
                return Some(InvariantViolation {
                    param: cmd.param.clone(),
                    kind: "unknown_device".into(),
                    detail: "no safe-range table for device (fail-closed)".into(),
                })
            }
        };
        // V2: alarm preservation — disabling/clearing an alarm parameter.
        if spec.alarm_params.iter().any(|p| p == &cmd.param) && cmd.value < 0.5 {
            return Some(InvariantViolation {
                param: cmd.param.clone(),
                kind: "alarm_preservation".into(),
                detail: "command would disable a device alarm/safety function".into(),
            });
        }
        // V3: unknown parameter for this device class.
        let range = match spec.params.get(&cmd.param) {
            Some(r) => r,
            None => {
                return Some(InvariantViolation {
                    param: cmd.param.clone(),
                    kind: "classification_integrity".into(),
                    detail: "parameter not part of this device's intended use".into(),
                })
            }
        };
        // V1: out-of-range.
        if cmd.value < range.min || cmd.value > range.max {
            return Some(InvariantViolation {
                param: cmd.param.clone(),
                kind: "out_of_range".into(),
                detail: format!("{} not in [{}, {}]", cmd.value, range.min, range.max),
            });
        }
        None
    }

    /// State-aware interlock check (e.g. ventilator PEEP < PIP). `state` is the
    /// device's settings *after* the candidate command would apply.
    pub fn check_interlock(
        &self,
        device: DeviceClass,
        state: &BTreeMap<String, f64>,
    ) -> Option<InvariantViolation> {
        let spec = self.spec(device)?;
        for il in &spec.interlocks {
            let (l, r) = (state.get(&il.left).copied(), state.get(&il.right).copied());
            if let (Some(l), Some(r)) = (l, r) {
                let ok = match il.op.as_str() {
                    "lt" => l < r,
                    "le" => l <= r,
                    _ => true,
                };
                if !ok {
                    return Some(InvariantViolation {
                        param: il.left.clone(),
                        kind: "interlock".into(),
                        detail: format!("interlock {} {} {} violated", il.left, il.op, il.right),
                    });
                }
            }
        }
        None
    }

    pub fn safe_state(&self, device: DeviceClass) -> BTreeMap<String, f64> {
        self.spec(device).map(|s| s.safe_state.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(device: DeviceClass, param: &str, value: f64) -> DeviceCommand {
        DeviceCommand { device, param: param.into(), value, patient_kg: Some(70.0) }
    }

    #[test]
    fn out_of_range_value_violates() {
        let set = InvariantSet::defaults();
        let v = set.check(&cmd(DeviceClass::Ventilator, "tidal_volume_ml", 1500.0));
        assert!(matches!(v, Some(InvariantViolation { ref kind, .. }) if kind == "out_of_range"));
    }

    #[test]
    fn in_range_value_ok() {
        let set = InvariantSet::defaults();
        assert!(set.check(&cmd(DeviceClass::Ventilator, "tidal_volume_ml", 450.0)).is_none());
    }

    #[test]
    fn disabling_alarm_violates() {
        let set = InvariantSet::defaults();
        let v = set.check(&cmd(DeviceClass::InfusionPump, "alarm_enabled", 0.0));
        assert!(matches!(v, Some(InvariantViolation { ref kind, .. }) if kind == "alarm_preservation"));
    }

    #[test]
    fn unknown_param_is_classification_violation() {
        let set = InvariantSet::defaults();
        let v = set.check(&cmd(DeviceClass::CardiacMonitor, "tidal_volume_ml", 500.0));
        assert!(matches!(v, Some(InvariantViolation { ref kind, .. }) if kind == "classification_integrity"));
    }

    #[test]
    fn interlock_peep_ge_pip_violates() {
        let set = InvariantSet::defaults();
        let mut state = BTreeMap::new();
        state.insert("peep_cmh2o".to_string(), 30.0);
        state.insert("pip_cmh2o".to_string(), 25.0);
        assert!(set.check_interlock(DeviceClass::Ventilator, &state).is_some());
    }

    #[test]
    fn default_table_matches_committed_json() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/datasets/014-device-broker/devicebench/devices.json"
        );
        let json = std::fs::read_to_string(path).expect("devices.json present");
        let from_file: InvariantSet = serde_json::from_str(&json).expect("parse devices.json");
        assert_eq!(from_file.version, InvariantSet::defaults().version);
        assert_eq!(from_file, InvariantSet::defaults());
    }
}
