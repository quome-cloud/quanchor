//! OpenAI adapter (`/v1/chat/completions`).
//!
//! Also serves OpenAI-compatible gateways (vLLM, LiteLLM, Together, etc.) via a
//! `base_url` override.

use super::{cost_from_table, Provider};
use crate::config::{ProviderKind, ProviderProfile};
use crate::ir::{LlmRequest, LlmResponse, Usage};
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct OpenAiProvider {
    client: reqwest::Client,
    name: String,
    base_url: String,
    api_key: Option<String>,
    passthrough: bool,
}

impl OpenAiProvider {
    pub fn new(client: reqwest::Client, profile: &ProviderProfile) -> Result<Self> {
        Ok(OpenAiProvider {
            client,
            name: profile.name.clone(),
            base_url: profile.effective_base_url(),
            api_key: profile.resolve_key(),
            passthrough: profile.is_passthrough(),
        })
    }
}

#[derive(Serialize)]
struct OaiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OaiRequest<'a> {
    model: &'a str,
    messages: Vec<OaiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    stream: bool,
}

#[derive(Deserialize)]
struct OaiResponse {
    #[serde(default)]
    model: String,
    choices: Vec<OaiChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OaiRespMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OaiRespMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize, Default)]
struct OaiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let mut messages: Vec<OaiMessage> = Vec::new();
        if let Some(sys) = &req.system {
            messages.push(OaiMessage { role: "system", content: sys });
        }
        for m in &req.messages {
            messages.push(OaiMessage { role: m.role.tag(), content: &m.content });
        }
        let body = OaiRequest {
            model: &req.model,
            messages,
            temperature: req.params.temperature,
            max_tokens: req.params.max_tokens,
            top_p: req.params.top_p,
            stream: false,
        };
        let key = self
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Provider("openai profile missing api_key".into()))?;
        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(Error::Http)?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "openai {status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let parsed: OaiResponse = resp.json().await.map_err(Error::Http)?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Provider("openai returned no choices".into()))?;
        let usage = parsed.usage.unwrap_or_default();
        let model = if parsed.model.is_empty() { req.model.clone() } else { parsed.model };
        let mut u = Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cost_usd: 0.0,
        };
        u.cost_usd = cost_from_table(ProviderKind::OpenAi, &model, &u);
        Ok(LlmResponse {
            content: choice.message.content.unwrap_or_default(),
            finish_reason: choice.finish_reason.unwrap_or_else(|| "stop".into()),
            usage: u,
            model,
        })
    }

    fn estimate_cost(&self, model: &str, usage: &Usage) -> f64 {
        cost_from_table(ProviderKind::OpenAi, model, usage)
    }
    fn auth_passthrough(&self) -> bool {
        self.passthrough
    }
}
