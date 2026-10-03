//! Ollama local API adapter (`/api/chat`).
//!
//! Ollama requires no API key and is the default provider for offline/local
//! benchmarking. Cost is always zero; token counts come from the response's
//! `prompt_eval_count` / `eval_count` fields when present.

use super::{estimate_tokens, Provider};
use crate::config::{ProviderKind, ProviderProfile};
use crate::ir::{LlmRequest, LlmResponse, Usage};
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct OllamaProvider {
    client: reqwest::Client,
    name: String,
    base_url: String,
    passthrough: bool,
}

impl OllamaProvider {
    pub fn new(client: reqwest::Client, profile: &ProviderProfile) -> Self {
        OllamaProvider {
            client,
            name: profile.name.clone(),
            base_url: profile.effective_base_url(),
            passthrough: profile.is_passthrough(),
        }
    }
}

#[derive(Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    #[serde(default)]
    model: String,
    message: Option<OllamaRespMessage>,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    done_reason: Option<String>,
}

#[derive(Deserialize)]
struct OllamaRespMessage {
    #[serde(default)]
    content: String,
}

#[async_trait]
impl Provider for OllamaProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let mut messages: Vec<OllamaMessage> = Vec::new();
        if let Some(sys) = &req.system {
            messages.push(OllamaMessage { role: "system", content: sys });
        }
        for m in &req.messages {
            messages.push(OllamaMessage { role: m.role.tag(), content: &m.content });
        }
        let body = OllamaChatRequest {
            model: &req.model,
            messages,
            stream: false,
            options: OllamaOptions {
                temperature: req.params.temperature,
                top_p: req.params.top_p,
                num_predict: req.params.max_tokens,
            },
        };
        let url = format!("{}/api/chat", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(Error::Http)?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "ollama {status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let parsed: OllamaChatResponse = resp.json().await.map_err(Error::Http)?;
        let content = parsed.message.map(|m| m.content).unwrap_or_default();
        let prompt_tokens = if parsed.prompt_eval_count > 0 {
            parsed.prompt_eval_count
        } else {
            estimate_tokens(&req.prompt_text())
        };
        let completion_tokens = if parsed.eval_count > 0 {
            parsed.eval_count
        } else {
            estimate_tokens(&content)
        };
        Ok(LlmResponse {
            model: if parsed.model.is_empty() { req.model.clone() } else { parsed.model },
            content,
            usage: Usage { prompt_tokens, completion_tokens, cost_usd: 0.0 },
            finish_reason: parsed.done_reason.unwrap_or_else(|| "stop".into()),
        })
    }

    fn estimate_cost(&self, _model: &str, _usage: &Usage) -> f64 {
        0.0
    }
    fn auth_passthrough(&self) -> bool {
        self.passthrough
    }
}
