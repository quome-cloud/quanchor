//! The real escalation primitive: park a request, notify a human channel,
//! resume on the answer, fail closed on timeout. Experiments use `OracleChannel`
//! (answers from TriageBench labels); production uses `WebhookChannel`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClinicianDecision {
    Allow,
    Deny,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationTicket {
    pub id: String,
    pub tool: String,
    pub risk: f64,
}

impl EscalationTicket {
    pub fn new(id: impl Into<String>, tool: impl Into<String>, risk: f64) -> Self {
        EscalationTicket {
            id: id.into(),
            tool: tool.into(),
            risk,
        }
    }
}

/// A human responder. `request` blocks until the clinician answers (or the impl
/// times out internally and returns `Timeout`).
#[async_trait::async_trait]
pub trait HumanChannel: Send + Sync {
    async fn request(&self, ticket: &EscalationTicket) -> ClinicianDecision;
}

/// Deterministic responder for experiments: returns a fixed/labelled decision.
pub struct OracleChannel {
    answer: ClinicianDecision,
    delay: Duration,
}

impl OracleChannel {
    pub fn allow() -> Self {
        OracleChannel {
            answer: ClinicianDecision::Allow,
            delay: Duration::ZERO,
        }
    }
    pub fn deny() -> Self {
        OracleChannel {
            answer: ClinicianDecision::Deny,
            delay: Duration::ZERO,
        }
    }
    pub fn answering(answer: ClinicianDecision, delay: Duration) -> Self {
        OracleChannel { answer, delay }
    }
}

#[async_trait::async_trait]
impl HumanChannel for OracleChannel {
    async fn request(&self, _t: &EscalationTicket) -> ClinicianDecision {
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        self.answer.clone()
    }
}

/// Holds parked escalations and enforces a timeout. `outstanding()` feeds the
/// load-aware throttle. A real deployment swaps `escalate` to register the
/// ticket and await an out-of-band resolve; the timeout path is identical.
pub struct TicketRegistry {
    timeout: Duration,
    outstanding: Arc<AtomicU32>,
}

impl TicketRegistry {
    pub fn new(timeout: Duration) -> Self {
        TicketRegistry {
            timeout,
            outstanding: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding.load(Ordering::SeqCst)
    }

    /// Park the ticket, ask the channel, resume on its answer — or fail closed
    /// (`Timeout`) if the human doesn't answer within the registry timeout.
    pub async fn escalate(
        &self,
        ticket: EscalationTicket,
        channel: &dyn HumanChannel,
    ) -> ClinicianDecision {
        self.outstanding.fetch_add(1, Ordering::SeqCst);
        let res = tokio::time::timeout(self.timeout, channel.request(&ticket)).await;
        self.outstanding.fetch_sub(1, Ordering::SeqCst);
        match res {
            Ok(decision) => decision,
            Err(_) => ClinicianDecision::Timeout,
        }
    }
}

/// One parked ticket's state. `Answered` buffers a resolution that arrived
/// before the gateway finished parking the request — so resolve-before-await is
/// race-free (a clinician may answer faster than the park completes).
enum Pending {
    Waiting(oneshot::Sender<ClinicianDecision>),
    Answered(ClinicianDecision),
}

/// Production transport. `request` registers a slot keyed by ticket id and (in a
/// real deploy) POSTs the ticket to the clinician system; the gateway's
/// `/oversight/resolve/{id}` endpoint calls `handle().resolve(...)` to wake it.
/// `in_memory()` omits the HTTP POST so it is unit-testable without a network.
pub struct WebhookChannel {
    inner: Arc<WebhookInner>,
}

struct WebhookInner {
    pending: Mutex<HashMap<String, Pending>>,
}

#[derive(Clone)]
pub struct WebhookHandle {
    inner: Arc<WebhookInner>,
}

impl WebhookChannel {
    pub fn in_memory() -> Self {
        WebhookChannel {
            inner: Arc::new(WebhookInner {
                pending: Mutex::new(HashMap::new()),
            }),
        }
    }
    pub fn handle(&self) -> WebhookHandle {
        WebhookHandle {
            inner: self.inner.clone(),
        }
    }
}

impl WebhookHandle {
    /// Deliver a clinician's answer to a parked ticket (called by the resolve
    /// endpoint). If the request hasn't finished parking yet, buffer the answer.
    pub fn resolve(&self, ticket_id: &str, decision: ClinicianDecision) {
        let mut pending = self.inner.pending.lock().unwrap();
        match pending.remove(ticket_id) {
            Some(Pending::Waiting(tx)) => {
                let _ = tx.send(decision);
            }
            _ => {
                pending.insert(ticket_id.to_string(), Pending::Answered(decision));
            }
        }
    }
}

#[async_trait::async_trait]
impl HumanChannel for WebhookChannel {
    async fn request(&self, t: &EscalationTicket) -> ClinicianDecision {
        let rx = {
            let mut pending = self.inner.pending.lock().unwrap();
            // A resolution may already be buffered (answered before we parked).
            if let Some(Pending::Answered(d)) = pending.remove(&t.id) {
                return d;
            }
            let (tx, rx) = oneshot::channel();
            pending.insert(t.id.clone(), Pending::Waiting(tx));
            rx
        };
        // real deploy: POST `t` to the clinician webhook here.
        rx.await.unwrap_or(ClinicianDecision::Timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn oracle_channel_answers_from_label() {
        let ch = OracleChannel::allow();
        let t = EscalationTicket::new("tkt-1", "order_medication", 0.4);
        assert_eq!(ch.request(&t).await, ClinicianDecision::Allow);
    }

    #[tokio::test]
    async fn escalate_resolves_via_channel() {
        let reg = TicketRegistry::new(Duration::from_secs(5));
        let t = EscalationTicket::new("tkt-2", "order_lab", 0.5);
        let got = reg.escalate(t, &OracleChannel::deny()).await;
        assert_eq!(got, ClinicianDecision::Deny);
    }

    #[tokio::test]
    async fn timeout_fails_closed() {
        let reg = TicketRegistry::new(Duration::from_millis(20));
        let t = EscalationTicket::new("tkt-3", "order_medication", 0.5);
        let slow = OracleChannel::answering(ClinicianDecision::Allow, Duration::from_millis(200));
        let got = reg.escalate(t, &slow).await;
        assert_eq!(got, ClinicianDecision::Timeout);
    }

    #[tokio::test]
    async fn webhook_channel_resolves_via_posted_answer() {
        let ch = WebhookChannel::in_memory();
        let handle = ch.handle();
        let t = EscalationTicket::new("tkt-4", "order_medication", 0.6);
        let fut = ch.request(&t);
        handle.resolve("tkt-4", ClinicianDecision::Allow);
        assert_eq!(fut.await, ClinicianDecision::Allow);
    }
}
