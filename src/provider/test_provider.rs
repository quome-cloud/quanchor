//! A deterministic, network-free `Provider` for tests: returns a canned (or
//! request-derived) response so the engine/proxy response path can be exercised
//! without a live model.

use super::{Provider, ProviderKind};
use crate::ir::{LlmRequest, LlmResponse, Usage};
use crate::Result;
use async_trait::async_trait;
use std::sync::Arc;

type Responder = Arc<dyn Fn(&LlmRequest) -> String + Send + Sync>;

pub struct TestProvider {
    name: String,
    responder: Responder,
}

impl TestProvider {
    /// A provider that always returns `text`.
    pub fn constant(name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        TestProvider { name: name.into(), responder: Arc::new(move |_| text.clone()) }
    }

    /// A provider whose response is computed from the request.
    pub fn from_fn(name: impl Into<String>, f: Responder) -> Self {
        TestProvider { name: name.into(), responder: f }
    }
}

#[async_trait]
impl Provider for TestProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn base_url(&self) -> &str {
        "test://local"
    }
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let content = (self.responder)(req);
        Ok(LlmResponse {
            model: req.model.clone(),
            content,
            usage: Usage::default(),
            finish_reason: "stop".to_string(),
        })
    }
    fn estimate_cost(&self, _model: &str, _usage: &Usage) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn constant_provider_returns_text() {
        let p = TestProvider::constant("t", "hello");
        let r = p.complete(&LlmRequest::user("m", "hi")).await.unwrap();
        assert_eq!(r.content, "hello");
    }
}
