//! Google Gemini adapter (`:generateContent`).

use super::{cost_from_table, Provider};
use crate::config::{ProviderKind, ProviderProfile};
use crate::ir::{LlmRequest, LlmResponse, Role, Usage};
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct GeminiProvider {
    client: reqwest::Client,
    name: String,
    base_url: String,
    api_key: Option<String>,
    passthrough: bool,
}

impl GeminiProvider {
    pub fn new(client: reqwest::Client, profile: &ProviderProfile) -> Result<Self> {
        Ok(GeminiProvider {
            client,
            name: profile.name.clone(),
            base_url: profile.effective_base_url(),
            api_key: profile.resolve_key(),
            passthrough: profile.is_passthrough(),
        })
    }
}

#[derive(Serialize)]
struct GemPart<'a> {
    text: &'a str,
}
#[derive(Serialize)]
struct GemContent<'a> {
    role: &'a str,
    parts: Vec<GemPart<'a>>,
}
#[derive(Serialize)]
struct GemSystem<'a> {
    parts: Vec<GemPart<'a>>,
}
#[derive(Serialize, Default)]
struct GemGenConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    top_p: Option<f64>,
}
#[derive(Serialize)]
struct GemRequest<'a> {
    contents: Vec<GemContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    system_instruction: Option<GemSystem<'a>>,
    #[serde(rename = "generationConfig")]
    generation_config: GemGenConfig,
}

#[derive(Deserialize)]
struct GemResponse {
    #[serde(default)]
    candidates: Vec<GemCandidate>,
    #[serde(default, rename = "usageMetadata")]
    usage_metadata: Option<GemUsage>,
}
#[derive(Deserialize)]
struct GemCandidate {
    #[serde(default)]
    content: Option<GemRespContent>,
    #[serde(default, rename = "finishReason")]
    finish_reason: Option<String>,
}
#[derive(Deserialize)]
struct GemRespContent {
    #[serde(default)]
    parts: Vec<GemRespPart>,
}
#[derive(Deserialize)]
struct GemRespPart {
    #[serde(default)]
    text: Option<String>,
}
#[derive(Deserialize, Default)]
struct GemUsage {
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: u32,
}

#[async_trait]
impl Provider for GeminiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Gemini
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let mut contents: Vec<GemContent> = Vec::new();
        for m in &req.messages {
            let role = match m.role {
                Role::Assistant => "model",
                _ => "user",
            };
            contents.push(GemContent { role, parts: vec![GemPart { text: &m.content }] });
        }
        let system_instruction = req
            .system
            .as_deref()
            .map(|s| GemSystem { parts: vec![GemPart { text: s }] });
        let body = GemRequest {
            contents,
            system_instruction,
            generation_config: GemGenConfig {
                temperature: req.params.temperature,
                max_output_tokens: req.params.max_tokens,
                top_p: req.params.top_p,
            },
        };
        let key = self
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Provider("gemini profile missing api_key".into()))?;
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, req.model, key
        );
        let resp = self.client.post(&url).json(&body).send().await.map_err(Error::Http)?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "gemini {status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let parsed: GemResponse = resp.json().await.map_err(Error::Http)?;
        let candidate = parsed
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| Error::Provider("gemini returned no candidates".into()))?;
        let content: String = candidate
            .content
            .map(|c| {
                c.parts
                    .into_iter()
                    .filter_map(|p| p.text)
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        let usage = parsed.usage_metadata.unwrap_or_default();
        let mut u = Usage {
            prompt_tokens: usage.prompt_token_count,
            completion_tokens: usage.candidates_token_count,
            cost_usd: 0.0,
        };
        u.cost_usd = cost_from_table(ProviderKind::Gemini, &req.model, &u);
        Ok(LlmResponse {
            content,
            finish_reason: candidate.finish_reason.unwrap_or_else(|| "STOP".into()),
            usage: u,
            model: req.model.clone(),
        })
    }

    fn estimate_cost(&self, model: &str, usage: &Usage) -> f64 {
        cost_from_table(ProviderKind::Gemini, model, usage)
    }
    fn auth_passthrough(&self) -> bool {
        self.passthrough
    }
}
