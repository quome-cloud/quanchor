//! Mandatory-confirmation gate for life-critical actuation (C8.3.4). Unlike the
//! 009 oversight router — which *decides whether* to escalate from a calibrated
//! risk — life-critical device confirmation is **mandatory, not risk-conditional**:
//! every such command is parked on a human. We therefore reuse 009's escalation
//! *transport* (`TicketRegistry` + `HumanChannel`, timeout ⇒ fail-closed) rather
//! than its risk router, which would (correctly, for information actions) auto-allow
//! a low-risk call.

use crate::oversight::channel::{ClinicianDecision, EscalationTicket, HumanChannel, TicketRegistry};
use std::time::Duration;

/// Default clinician-confirmation timeout. Past this, fail closed (deny).
pub const DEFAULT_CONFIRM_TIMEOUT: Duration = Duration::from_secs(30);

pub struct ConfirmGate {
    registry: TicketRegistry,
}

impl ConfirmGate {
    pub fn new(timeout: Duration) -> Self {
        ConfirmGate { registry: TicketRegistry::new(timeout) }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_CONFIRM_TIMEOUT)
    }

    pub fn outstanding(&self) -> u32 {
        self.registry.outstanding()
    }

    /// Park a life-critical actuation on the human channel and return their
    /// decision. `Timeout` is surfaced for the audit trail; callers treat
    /// anything other than `Allow` as a denial.
    pub async fn confirm(
        &self,
        action_id: &str,
        tool: &str,
        channel: &dyn HumanChannel,
    ) -> ClinicianDecision {
        // risk=1.0: a life-critical actuation is treated as maximal-risk on the
        // ticket so the bedside UI surfaces it accordingly.
        let ticket = EscalationTicket::new(action_id, tool, 1.0);
        self.registry.escalate(ticket, channel).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oversight::channel::OracleChannel;

    #[tokio::test]
    async fn life_critical_in_range_requires_human_allow() {
        let gate = ConfirmGate::with_defaults();
        let d = gate.confirm("act-1", "ventilator", &OracleChannel::allow()).await;
        assert_eq!(d, ClinicianDecision::Allow);
    }

    #[tokio::test]
    async fn human_deny_blocks() {
        let gate = ConfirmGate::with_defaults();
        assert_eq!(
            gate.confirm("a", "ventilator", &OracleChannel::deny()).await,
            ClinicianDecision::Deny
        );
    }

    #[tokio::test]
    async fn timeout_fails_closed() {
        // 1ms timeout, channel that answers after 50ms -> Timeout (fail closed).
        let gate = ConfirmGate::new(Duration::from_millis(1));
        let slow = OracleChannel::answering(ClinicianDecision::Allow, Duration::from_millis(50));
        assert_eq!(gate.confirm("a", "ventilator", &slow).await, ClinicianDecision::Timeout);
    }
}
