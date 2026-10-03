//! Anthropic Messages API adapter (`/v1/messages`).

use super::{cost_from_table, Provider};
use crate::config::{ProviderKind, ProviderProfile};
use crate::ir::{LlmRequest, LlmResponse, Role, Usage};
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct AnthropicProvider {
    client: reqwest::Client,
    name: String,
    base_url: String,
    api_key: Option<String>,
    passthrough: bool,
}

impl AnthropicProvider {
    pub fn new(client: reqwest::Client, profile: &ProviderProfile) -> Result<Self> {
        Ok(AnthropicProvider {
            client,
            name: profile.name.clone(),
            base_url: profile.effective_base_url(),
            api_key: profile.resolve_key(),
            passthrough: profile.is_passthrough(),
        })
    }
}

#[derive(Serialize)]
struct AntMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct AntRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<AntMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Deserialize)]
struct AntResponse {
    #[serde(default)]
    model: String,
    #[serde(default)]
    content: Vec<AntBlock>,
    #[serde(default)]
    usage: Option<AntUsage>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct AntBlock {
    #[serde(default, rename = "type")]
    _kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize, Default)]
struct AntUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        // Anthropic only accepts user/assistant turns; any IR system message is
        // hoisted into the top-level `system` field.
        let mut system = req.system.clone();
        let mut messages: Vec<AntMessage> = Vec::new();
        for m in &req.messages {
            match m.role {
                Role::System => {
                    // Merge stray system turns into the system field.
                    system = Some(match system {
                        Some(s) => format!("{s}\n{}", m.content),
                        None => m.content.clone(),
                    });
                }
                Role::User | Role::Tool => {
                    messages.push(AntMessage { role: "user", content: &m.content })
                }
                Role::Assistant => {
                    messages.push(AntMessage { role: "assistant", content: &m.content })
                }
            }
        }
        let body = AntRequest {
            model: &req.model,
            max_tokens: req.params.max_tokens.unwrap_or(1024),
            system: system.as_deref(),
            messages,
            temperature: req.params.temperature,
        };
        let key = self
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Provider("anthropic profile missing api_key".into()))?;
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(Error::Http)?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "anthropic {status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let parsed: AntResponse = resp.json().await.map_err(Error::Http)?;
        let content: String = parsed
            .content
            .into_iter()
            .filter_map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");
        let usage = parsed.usage.unwrap_or_default();
        let model = if parsed.model.is_empty() { req.model.clone() } else { parsed.model };
        let mut u = Usage {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            cost_usd: 0.0,
        };
        u.cost_usd = cost_from_table(ProviderKind::Anthropic, &model, &u);
        Ok(LlmResponse {
            content,
            finish_reason: parsed.stop_reason.unwrap_or_else(|| "end_turn".into()),
            usage: u,
            model,
        })
    }

    fn estimate_cost(&self, model: &str, usage: &Usage) -> f64 {
        cost_from_table(ProviderKind::Anthropic, model, usage)
    }
    fn auth_passthrough(&self) -> bool {
        self.passthrough
    }
}
