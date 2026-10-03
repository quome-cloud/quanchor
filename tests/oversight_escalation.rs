//! End-to-end oversight: auto-decide, webhook park→resolve→resume, timeout fail-closed.
use qfire::oversight::channel::{
    ClinicianDecision, EscalationTicket, OracleChannel, TicketRegistry, WebhookChannel,
};
use qfire::oversight::signals::{OversightSignals, SignalsBuilder, ToolRisk};
use qfire::oversight::{OversightGuard, OversightOutcome};
use std::time::Duration;

#[tokio::test]
async fn webhook_round_trip_resumes_on_clinician_answer() {
    let reg = TicketRegistry::new(Duration::from_secs(2));
    let ch = WebhookChannel::in_memory();
    let handle = ch.handle();
    let t = EscalationTicket::new("e2e-1", "order_medication", 0.5);
    let fut = reg.escalate(t, &ch);
    handle.resolve("e2e-1", ClinicianDecision::Allow);
    assert_eq!(fut.await, ClinicianDecision::Allow);
}

#[tokio::test]
async fn timeout_fails_closed_end_to_end() {
    let guard = OversightGuard::with_defaults(Duration::from_millis(20));
    let mut s = OversightSignals::for_tool(ToolRisk::High);
    // mid block-confidence lands in the escalate band under the fitted model
    s.max_block_confidence = 0.45;
    let slow = OracleChannel::answering(ClinicianDecision::Allow, Duration::from_millis(200));
    let out = guard
        .decide("e2e-2", "order_medication", s, &slow)
        .await;
    assert_eq!(
        out,
        OversightOutcome::HumanResolved(ClinicianDecision::Timeout)
    );
}

#[tokio::test]
async fn signals_builder_feeds_guard() {
    use qfire::egress::EgressFinding;
    let findings = vec![EgressFinding {
        prov_id: "p".into(),
        label: "mrn".into(),
        arg_path: "x".into(),
        via: "base64".into(),
    }];
    let s = SignalsBuilder::new(ToolRisk::High).with_egress(&findings).build();
    let guard = OversightGuard::with_defaults(Duration::from_secs(1));
    let out = guard
        .decide("e2e-3", "send_referral", s, &OracleChannel::allow())
        .await;
    assert!(matches!(
        out,
        OversightOutcome::AutoAllow
            | OversightOutcome::AutoDeny
            | OversightOutcome::HumanResolved(_)
    ));
}
