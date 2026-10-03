//! Provider profiles and runtime configuration.
//!
//! Configuration is plain TOML, loaded from (in order) an explicit `--config`
//! path, `./qfire.toml`, or `~/.config/qfire/config.toml`. If no file is found,
//! a built-in default profile pointing at local Ollama is used so QFIRE works
//! offline with zero setup.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A provider family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    #[serde(alias = "openai")]
    OpenAi,
    Anthropic,
    Gemini,
    Ollama,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::OpenAi => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Gemini => "gemini",
            ProviderKind::Ollama => "ollama",
        }
    }

    /// The conventional default base URL for this provider family.
    pub fn default_base_url(self) -> &'static str {
        match self {
            ProviderKind::OpenAi => "https://api.openai.com",
            ProviderKind::Anthropic => "https://api.anthropic.com",
            ProviderKind::Gemini => "https://generativelanguage.googleapis.com",
            ProviderKind::Ollama => "http://localhost:11434",
        }
    }
}

/// A named provider profile: credentials + base URL for one downstream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderProfile {
    pub name: String,
    pub kind: ProviderKind,
    /// Base URL override; defaults to the family default.
    #[serde(default)]
    pub base_url: Option<String>,
    /// API key, or the name of an env var to read it from (`env:OPENAI_API_KEY`).
    #[serde(default)]
    pub api_key: Option<String>,
    /// A default model for this profile (used when a request omits one).
    #[serde(default)]
    pub model: Option<String>,
}

impl ProviderProfile {
    /// The effective base URL.
    pub fn effective_base_url(&self) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| self.kind.default_base_url().to_string())
    }

    /// True when this profile forwards the caller's provider credentials
    /// downstream instead of injecting a configured key.
    pub fn is_passthrough(&self) -> bool {
        matches!(self.api_key.as_deref(), Some("passthrough"))
    }

    /// Resolve the API key, expanding an `env:VAR` reference.
    pub fn resolve_key(&self) -> Option<String> {
        match &self.api_key {
            None => None,
            Some(k) if k == "passthrough" => None,
            Some(k) => {
                if let Some(var) = k.strip_prefix("env:") {
                    std::env::var(var).ok()
                } else {
                    Some(k.clone())
                }
            }
        }
    }
}

/// Proxy server settings (inbound auth).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCfg {
    /// Required inbound auth token for `qfire serve` (checked against the
    /// `X-QFire-Token` request header). `env:VAR` expands from the
    /// environment. Unset disables the check (local dev).
    #[serde(default)]
    pub auth_token: Option<String>,
}

impl ServerCfg {
    /// Resolve the token, expanding an `env:VAR` reference.
    pub fn resolve_token(&self) -> Option<String> {
        match &self.auth_token {
            None => None,
            Some(k) => match k.strip_prefix("env:") {
                Some(var) => std::env::var(var).ok(),
                None => Some(k.clone()),
            },
        }
    }
}

/// Top-level QFIRE configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Provider profiles. The first listed is the default.
    #[serde(default)]
    pub providers: Vec<ProviderProfile>,
    /// Directory holding the YAML rule library.
    #[serde(default = "default_rules_dir")]
    pub rules_dir: String,
    /// Directory holding chain definitions.
    #[serde(default = "default_chains_dir")]
    pub chains_dir: String,
    /// Path to the append-only audit log.
    #[serde(default = "default_audit_path")]
    pub audit_path: String,
    /// 003 tamper-evident audit settings.
    #[serde(default)]
    pub audit: AuditCfg,
    /// 004 supply-chain admission settings.
    #[serde(default)]
    pub admission: crate::admission::AdmissionCfg,
    /// 008 agent-identity settings.
    #[serde(default)]
    pub identity: crate::identity::IdentityCfg,
    /// Proxy server settings (inbound auth for `qfire serve`).
    #[serde(default)]
    pub server: ServerCfg,
}

/// 003 tamper-evident audit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditCfg {
    /// plain | chained | chained_signed | chained_signed_batched
    pub mode: crate::audit::store::Mode,
    /// ed25519 keyfile (hex seed). Generated on first use if absent.
    pub key_path: String,
    /// Anchors JSONL path ("" disables anchoring).
    pub anchors_path: String,
    /// Publish an anchor every k entries (0 disables).
    pub anchor_every: u64,
    /// Checkpoint cadence in batched mode.
    pub batch: u64,
    /// If true, audit failures do NOT fail the call (default false: fail-closed).
    pub fail_open: bool,
}

impl Default for AuditCfg {
    fn default() -> Self {
        AuditCfg {
            mode: crate::audit::store::Mode::ChainedSigned,
            key_path: "audit.key".into(),
            anchors_path: String::new(),
            anchor_every: 0,
            batch: 64,
            fail_open: false,
        }
    }
}

fn default_rules_dir() -> String {
    "rules".into()
}
fn default_chains_dir() -> String {
    "chains".into()
}
fn default_audit_path() -> String {
    "audit.jsonl".into()
}

impl Default for Config {
    fn default() -> Self {
        Config {
            providers: vec![ProviderProfile {
                name: "ollama".into(),
                kind: ProviderKind::Ollama,
                base_url: None,
                api_key: None,
                model: Some("llama3.2".into()),
            }],
            rules_dir: default_rules_dir(),
            chains_dir: default_chains_dir(),
            audit_path: default_audit_path(),
            audit: AuditCfg::default(),
            admission: crate::admission::AdmissionCfg::default(),
            identity: crate::identity::IdentityCfg::default(),
            server: ServerCfg::default(),
        }
    }
}

impl Config {
    /// Load configuration, searching standard locations when `explicit` is None.
    /// Falls back to a built-in Ollama-only default if nothing is found.
    pub fn load(explicit: Option<&Path>) -> Result<Config> {
        let candidates: Vec<std::path::PathBuf> = match explicit {
            Some(p) => vec![p.to_path_buf()],
            None => {
                let mut v = vec![std::path::PathBuf::from("qfire.toml")];
                if let Some(home) = dirs::config_dir() {
                    v.push(home.join("qfire").join("config.toml"));
                }
                v
            }
        };
        for path in candidates {
            if path.exists() {
                let text = std::fs::read_to_string(&path)?;
                let cfg: Config = toml::from_str(&text)
                    .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
                return Ok(cfg);
            }
        }
        if let Some(p) = explicit {
            return Err(Error::Config(format!("config not found: {}", p.display())));
        }
        Ok(Config::default())
    }

    /// Look up a profile by name.
    pub fn profile(&self, name: &str) -> Option<&ProviderProfile> {
        self.providers.iter().find(|p| p.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_cfg_resolves_literal_token() {
        let cfg: Config = toml::from_str("[server]\nauth_token = \"sekrit\"\n").unwrap();
        assert_eq!(cfg.server.resolve_token().as_deref(), Some("sekrit"));
    }

    #[test]
    fn server_cfg_resolves_env_token() {
        std::env::set_var("QFIRE_TEST_AUTH_TOKEN_A1", "from-env");
        let cfg: Config =
            toml::from_str("[server]\nauth_token = \"env:QFIRE_TEST_AUTH_TOKEN_A1\"\n").unwrap();
        assert_eq!(cfg.server.resolve_token().as_deref(), Some("from-env"));
    }

    #[test]
    fn server_cfg_defaults_to_no_token() {
        let cfg: Config = toml::from_str("").unwrap();
        assert!(cfg.server.resolve_token().is_none());
    }

    #[test]
    fn server_cfg_env_unset_resolves_none() {
        let cfg: Config =
            toml::from_str("[server]\nauth_token = \"env:QFIRE_TEST_UNSET_VAR_A1F\"\n").unwrap();
        assert!(cfg.server.resolve_token().is_none());
        assert!(cfg.server.auth_token.is_some()); // configured-but-unresolvable is detectable
    }

    #[test]
    fn passthrough_profile_detected_and_never_resolves_a_key() {
        let p = ProviderProfile {
            name: "openai".into(),
            kind: ProviderKind::OpenAi,
            base_url: None,
            api_key: Some("passthrough".into()),
            model: None,
        };
        assert!(p.is_passthrough());
        assert_eq!(p.resolve_key(), None);
        let lit = ProviderProfile { api_key: Some("sk-real".into()), ..p.clone() };
        assert!(!lit.is_passthrough());
        assert_eq!(lit.resolve_key().as_deref(), Some("sk-real"));
    }

    #[test]
    fn provider_kind_accepts_openai_spelling() {
        let cfg: Config = toml::from_str(
            "[[providers]]\nname = \"o\"\nkind = \"openai\"\napi_key = \"passthrough\"\n",
        )
        .unwrap();
        assert_eq!(cfg.providers[0].kind, ProviderKind::OpenAi);
        // canonical snake_case still accepted
        let cfg2: Config = toml::from_str(
            "[[providers]]\nname = \"o\"\nkind = \"open_ai\"\napi_key = \"passthrough\"\n",
        )
        .unwrap();
        assert_eq!(cfg2.providers[0].kind, ProviderKind::OpenAi);
    }
}
