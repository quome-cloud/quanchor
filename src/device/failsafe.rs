//! Fail-safe controller (C8.3.5, C3.1). A heartbeat tracks liveness between the
//! gateway and the device; on loss beyond a grace window the controller drives
//! the device to its documented safe state rather than leaving the last command
//! latched. Crucially this logic is **local to the broker**, so it holds even
//! when the LLM/agent is unreachable — that is the whole point of resilience.

use super::sim::DeviceSim;
use serde::Serialize;

/// What to do when the agent/network partitions mid-actuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionPolicy {
    /// Hold the last command. The naive, *unsafe* baseline.
    Freeze,
    /// Drive the device to its documented safe state. Recommended.
    RevertToSafe,
    /// Revert to safe state AND raise a handoff alarm for the bedside human.
    Handoff,
}

/// The action the controller took (for E2/E5 scoring + the audit trail).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailsafeAction {
    HeldLastCommand,
    RevertedToSafe,
    HandedOff,
}

/// Liveness tracker: the device is considered lost once `missed` consecutive
/// heartbeats exceed the grace window.
#[derive(Debug, Clone, Copy)]
pub struct Heartbeat {
    grace: u32,
}

impl Heartbeat {
    pub fn new(grace: u32) -> Self {
        Heartbeat { grace }
    }
    pub fn lost(&self, missed: u32) -> bool {
        missed > self.grace
    }
}

pub struct FailsafeController {
    policy: PartitionPolicy,
}

impl FailsafeController {
    pub fn new(policy: PartitionPolicy) -> Self {
        FailsafeController { policy }
    }

    /// Apply the partition policy to a device that just lost its agent.
    pub fn on_partition(&self, sim: &mut DeviceSim) -> FailsafeAction {
        match self.policy {
            PartitionPolicy::Freeze => FailsafeAction::HeldLastCommand,
            PartitionPolicy::RevertToSafe => {
                sim.reset_to_safe();
                FailsafeAction::RevertedToSafe
            }
            PartitionPolicy::Handoff => {
                sim.reset_to_safe();
                FailsafeAction::HandedOff
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceClass, DeviceCommand};

    fn vent(param: &str, value: f64) -> DeviceCommand {
        DeviceCommand { device: DeviceClass::Ventilator, param: param.into(), value, patient_kg: Some(70.0) }
    }

    #[test]
    fn revert_to_safe_drives_device_to_safe_state_on_partition() {
        let mut v = DeviceSim::ventilator();
        v.apply(&vent("tidal_volume_ml", 1200.0)); // unsafe, latched
        assert!(v.is_unsafe());
        let ctrl = FailsafeController::new(PartitionPolicy::RevertToSafe);
        assert_eq!(ctrl.on_partition(&mut v), FailsafeAction::RevertedToSafe);
        assert!(!v.is_unsafe(), "must hold safe state, not latch last command");
    }

    #[test]
    fn freeze_latches_last_command() {
        let mut v = DeviceSim::ventilator();
        v.apply(&vent("tidal_volume_ml", 1200.0));
        let ctrl = FailsafeController::new(PartitionPolicy::Freeze);
        assert_eq!(ctrl.on_partition(&mut v), FailsafeAction::HeldLastCommand);
        assert!(v.is_unsafe(), "freeze is the unsafe baseline");
    }

    #[test]
    fn heartbeat_loss_detected_after_grace() {
        let hb = Heartbeat::new(2);
        assert!(!hb.lost(1));
        assert!(!hb.lost(2));
        assert!(hb.lost(3));
    }
}
