//! Paper 011 — off-hot-path continuous performance & drift monitoring.
//!
//! The monitor consumes a [`MonitorEvent`] stream — the read-side view of the
//! 003 audit log, joined out-of-band with clinical outcome, autonomy, and
//! behavioral fields (see the feature spec, §3/§7). Detectors are sequential and
//! memory-bounded; they never touch the request path, so the layer adds zero
//! per-call latency. HAARF-Drift emits [`MonitorEvent`]s directly.

pub mod alert;
pub mod autonomy;
pub mod behavior;
pub mod drift;
pub mod rwe;

use serde::{Deserialize, Serialize};

/// One decision the monitor observes, off the hot path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorEvent {
    /// Monotonically increasing case index within a stream.
    pub case: u64,
    /// RFC3339 timestamp (from `AuditRecord.ts`); optional for synthetic streams.
    #[serde(default)]
    pub ts: Option<String>,
    /// Agent identity (008) for behavioral profiling.
    #[serde(default)]
    pub agent_id: String,
    /// Tool invoked (behavioral profiling).
    #[serde(default)]
    pub tool: String,
    /// Terminal verdict string from `AuditRecord.terminal`.
    #[serde(default)]
    pub verdict: String,
    /// Max node injection/block score in [0,1] (the proxy signal).
    #[serde(default)]
    pub score: f64,
    /// Lagged clinical outcome: `Some(true)`=good, `Some(false)`=bad, `None`=not yet known (D4).
    #[serde(default)]
    pub outcome: Option<bool>,
    /// Risk tier of the action actually taken (D2).
    #[serde(default)]
    pub autonomy_level: u8,
    /// Whether the action was taken autonomously (no human in the loop).
    #[serde(default)]
    pub autonomous: bool,
}

/// Emitted by a detector when it raises an alarm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftSignal {
    pub detector: String,
    pub case: u64,
    /// Detector-specific statistic at the alarm.
    pub statistic: f64,
    /// Normalized severity in [0,1], consumed by the alert ladder.
    pub severity: f64,
}

/// A sequential, memory-bounded change detector over a scalar proxy signal.
pub trait StreamDetector {
    fn name(&self) -> &str;
    /// Observe one scalar for `case`; return `Some` on alarm.
    fn observe(&mut self, case: u64, x: f64) -> Option<DriftSignal>;
    /// Reset to initial state (for reuse across streams).
    fn reset(&mut self);
}

/// Graduated response level (models the 009 oversight-router contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertLevel {
    Log,
    Notify,
    Throttle,
    Halt,
}

/// A graduated alert raised from a [`DriftSignal`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub level: AlertLevel,
    pub case: u64,
    pub source: String,
    pub detail: String,
}
