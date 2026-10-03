//! The unified provider client.
//!
//! One trait — [`Provider`] — abstracts over OpenAI (chat/completions +
//! responses), Anthropic Messages, Google Gemini `generateContent`, and the
//! Ollama local API. Adapters normalize requests and responses to the crate
//! [`crate::ir`] so a firewall chain is provider-independent. Credentials and
//! base URLs are configured per [`crate::config::ProviderProfile`]; Ollama needs
//! no key and is the default for offline/local benchmarking.

mod anthropic;
mod gemini;
mod ollama;
mod openai;
pub mod test_provider;

pub use anthropic::AnthropicProvider;
pub use gemini::GeminiProvider;
pub use ollama::OllamaProvider;
pub use openai::OpenAiProvider;
pub use test_provider::TestProvider;

use crate::config::{ProviderKind, ProviderProfile};
use crate::ir::{LlmRequest, LlmResponse, Usage};
use crate::{Error, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// A downstream LLM provider.
#[async_trait]
pub trait Provider: Send + Sync {
    /// The provider family.
    fn kind(&self) -> ProviderKind;
    /// The profile name this provider was instantiated from.
    fn name(&self) -> &str;
    /// The configured base URL (used by the proxy for raw forwarding).
    fn base_url(&self) -> &str;
    /// A version/identity string for reproducibility (kind + base host).
    fn version(&self) -> String {
        format!("{}@{}", self.kind().as_str(), self.base_url())
    }
    /// Issue a non-streaming completion, returning the normalized response.
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse>;
    /// Estimate the USD cost of a call from token usage. Local providers (Ollama)
    /// return `0.0`.
    fn estimate_cost(&self, model: &str, usage: &Usage) -> f64;
    /// True when the proxy should forward the caller's provider credentials
    /// downstream (profile configured with `api_key = "passthrough"`).
    fn auth_passthrough(&self) -> bool {
        false
    }
}

/// A registry of configured providers, keyed by profile name.
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
    default: Option<String>,
}

impl ProviderRegistry {
    /// Build a registry from a set of profiles, sharing one HTTP client.
    pub fn from_profiles(profiles: &[ProviderProfile]) -> Result<Self> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(Error::Http)?;
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        let mut default = None;
        for p in profiles {
            if default.is_none() {
                default = Some(p.name.clone());
            }
            let prov: Arc<dyn Provider> = match p.kind {
                ProviderKind::Ollama => Arc::new(OllamaProvider::new(client.clone(), p)),
                ProviderKind::OpenAi => Arc::new(OpenAiProvider::new(client.clone(), p)?),
                ProviderKind::Anthropic => Arc::new(AnthropicProvider::new(client.clone(), p)?),
                ProviderKind::Gemini => Arc::new(GeminiProvider::new(client.clone(), p)?),
            };
            providers.insert(p.name.clone(), prov);
        }
        Ok(ProviderRegistry { providers, default })
    }

    /// Look up a provider by profile name.
    pub fn get(&self, name: &str) -> Result<Arc<dyn Provider>> {
        self.providers
            .get(name)
            .cloned()
            .ok_or_else(|| Error::Provider(format!("no provider profile named '{name}'")))
    }

    /// The default provider (first configured), if any.
    pub fn default(&self) -> Result<Arc<dyn Provider>> {
        let name = self
            .default
            .as_ref()
            .ok_or_else(|| Error::Provider("no providers configured".into()))?;
        self.get(name)
    }

    /// The name of the default provider, if any.
    pub fn default_name(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// All configured profile names.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.providers.keys().cloned().collect();
        v.sort();
        v
    }
}

/// A small, citable pricing table (USD per 1M tokens, prompt/completion).
///
/// Prices are approximate and exist so the benchmark can quantify firewall
/// overhead. They are versioned with the source date in the manifest.
pub fn price_per_mtok(kind: ProviderKind, model: &str) -> (f64, f64) {
    let m = model.to_ascii_lowercase();
    match kind {
        ProviderKind::Ollama => (0.0, 0.0),
        ProviderKind::OpenAi => {
            if m.contains("gpt-4o-mini") {
                (0.15, 0.60)
            } else if m.contains("gpt-4o") {
                (2.50, 10.0)
            } else if m.contains("o1") || m.contains("o3") {
                (15.0, 60.0)
            } else {
                (0.50, 1.50)
            }
        }
        ProviderKind::Anthropic => {
            if m.contains("haiku") {
                (0.80, 4.0)
            } else if m.contains("opus") {
                (15.0, 75.0)
            } else {
                (3.0, 15.0) // sonnet-class default
            }
        }
        ProviderKind::Gemini => {
            if m.contains("flash") {
                (0.075, 0.30)
            } else {
                (1.25, 5.0) // pro-class default
            }
        }
    }
}

/// Compute a cost estimate from a price table and usage.
pub fn cost_from_table(kind: ProviderKind, model: &str, usage: &Usage) -> f64 {
    let (pin, pout) = price_per_mtok(kind, model);
    (usage.prompt_tokens as f64 / 1_000_000.0) * pin
        + (usage.completion_tokens as f64 / 1_000_000.0) * pout
}

/// A rough token estimate used when a provider does not report usage (e.g. some
/// Ollama responses): ~4 characters per token.
pub fn estimate_tokens(text: &str) -> u32 {
    ((text.chars().count() as f64) / 4.0).ceil() as u32
}
