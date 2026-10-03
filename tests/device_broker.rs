//! End-to-end tests for the 014 device-safety broker, exercising only the public
//! API (`qfire::device::*`). These assert the headline safety properties:
//! unsafe actuation is denied broker-on but reaches the raw device broker-off
//! (V1), life-critical actuation without a human is blocked (V5), and a
//! partition with RevertToSafe holds the device's safe state (V4/C3.1).

use qfire::device::failsafe::{FailsafeController, PartitionPolicy};
use qfire::device::sim::DeviceSim;
use qfire::device::{DeviceClass, DeviceCommand, DeviceGuard, DeviceOutcome};
use qfire::oversight::channel::OracleChannel;

fn cmd(device: DeviceClass, param: &str, value: f64) -> DeviceCommand {
    DeviceCommand { device, param: param.into(), value, patient_kg: Some(70.0) }
}

#[tokio::test]
async fn unsafe_command_denied_on_but_reaches_device_off() {
    let guard = DeviceGuard::with_defaults();
    let dangerous = cmd(DeviceClass::Ventilator, "tidal_volume_ml", 1500.0);

    // broker-off: the raw device accepts the dangerous command and is unsafe.
    let mut off = DeviceSim::ventilator();
    off.apply(&dangerous);
    assert!(off.is_unsafe(), "without the broker the device is driven unsafe");

    // broker-on: denied; the command never reaches the device.
    let mut on = DeviceSim::ventilator();
    let out = guard.check(&dangerous, &OracleChannel::deny()).await;
    assert!(matches!(out, DeviceOutcome::Deny { .. }));
    if matches!(out, DeviceOutcome::Allow | DeviceOutcome::Clamp { .. }) {
        on.apply(&dangerous);
    }
    assert!(!on.is_unsafe(), "with the broker the device stays safe");
}

#[tokio::test]
async fn life_critical_without_human_is_blocked() {
    let guard = DeviceGuard::with_defaults();
    // An in-range, life-critical command with no human available (deny channel
    // models "no confirmation") must be blocked — V5 eliminated.
    let in_range = cmd(DeviceClass::InfusionPump, "rate_ml_per_hr", 50.0);
    let out = guard.check(&in_range, &OracleChannel::deny()).await;
    assert!(matches!(out, DeviceOutcome::Deny { .. }));

    // The same command WITH a human confirmation is allowed.
    let ok = guard.check(&in_range, &OracleChannel::allow()).await;
    assert_eq!(ok, DeviceOutcome::Allow);
}

#[tokio::test]
async fn partition_with_revert_holds_safe_state() {
    let mut v = DeviceSim::ventilator();
    v.apply(&cmd(DeviceClass::Ventilator, "fio2", 1.0)); // borderline-high, latched
    v.apply(&cmd(DeviceClass::Ventilator, "tidal_volume_ml", 1200.0)); // unsafe, latched
    assert!(v.is_unsafe());

    FailsafeController::new(PartitionPolicy::RevertToSafe).on_partition(&mut v);
    assert!(!v.is_unsafe(), "RevertToSafe must hold the documented safe state");
    assert!(v.alarms_enabled());
}
