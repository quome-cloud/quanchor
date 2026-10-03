//! Risk-calibrated human oversight (paper 009). The gateway's last decision
//! layer: aggregates the confidences/deny-reasons of all prior layers into a
//! calibrated risk, then routes auto-allow / auto-deny / escalate-to-human.
//! `Verdict` (the detector vocabulary) is untouched; escalation lives here.

pub mod channel;
pub mod router;
pub mod scorer;
pub mod signals;
pub mod throttle;

use std::time::Duration;

use crate::oversight::channel::{ClinicianDecision, EscalationTicket, HumanChannel, TicketRegistry};
use crate::oversight::router::{route, Load, RouteClass, Thresholds};
use crate::oversight::scorer::{Calibration, RiskScorer};
use crate::oversight::signals::OversightSignals;

/// The terminal oversight decision. Distinct from `Verdict` (the detector
/// vocabulary): this is the routing layer's outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OversightOutcome {
    AutoAllow,
    AutoDeny,
    /// Escalated to a human and resolved with their decision (`Timeout` ⇒ treat
    /// as deny by the caller; surfaced explicitly for the audit trail).
    HumanResolved(ClinicianDecision),
}

/// The last decision layer before execution.
pub struct OversightGuard {
    scorer: RiskScorer,
    thresholds: Thresholds,
    registry: TicketRegistry,
}

impl OversightGuard {
    pub fn new(cal: Calibration, thresholds: Thresholds, timeout: Duration) -> Self {
        OversightGuard {
            scorer: RiskScorer::new(cal),
            thresholds,
            registry: TicketRegistry::new(timeout),
        }
    }

    pub fn with_defaults(timeout: Duration) -> Self {
        Self::new(Calibration::default(), Thresholds::default(), timeout)
    }

    pub fn outstanding(&self) -> u32 {
        self.registry.outstanding()
    }

    /// Score → route → (maybe) escalate. Pure auto-decisions never touch the
    /// channel; escalations park on it and resume on the human's answer.
    pub async fn decide(
        &self,
        action_id: &str,
        tool: &str,
        signals: OversightSignals,
        channel: &dyn HumanChannel,
    ) -> OversightOutcome {
        let risk = self.scorer.score(&signals);
        let load = Load {
            outstanding: self.registry.outstanding(),
        };
        match route(risk, signals.tool_risk, load, &self.thresholds) {
            RouteClass::AutoAllow => OversightOutcome::AutoAllow,
            RouteClass::AutoDeny => OversightOutcome::AutoDeny,
            RouteClass::Escalate => {
                let ticket = EscalationTicket::new(action_id, tool, risk);
                let decision = self.registry.escalate(ticket, channel).await;
                OversightOutcome::HumanResolved(decision)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oversight::channel::{ClinicianDecision, OracleChannel};
    use crate::oversight::signals::{OversightSignals, ToolRisk};

    #[tokio::test]
    async fn clean_low_risk_auto_allows_without_human() {
        let guard = OversightGuard::with_defaults(Duration::from_secs(1));
        let s = OversightSignals::for_tool(ToolRisk::Low);
        let out = guard
            .decide("act-1", "read_note", s, &OracleChannel::deny())
            .await;
        assert_eq!(out, OversightOutcome::AutoAllow); // never asked the human
    }

    #[tokio::test]
    async fn ambiguous_action_escalates_and_takes_human_answer() {
        let guard = OversightGuard::with_defaults(Duration::from_secs(1));
        let mut s = OversightSignals::for_tool(ToolRisk::High);
        s.policy_margin = Some(0.5); // policy escalated
        s.max_block_confidence = 0.5;
        let out = guard
            .decide("act-2", "order_medication", s, &OracleChannel::deny())
            .await;
        assert_eq!(
            out,
            OversightOutcome::HumanResolved(ClinicianDecision::Deny)
        );
    }
}
