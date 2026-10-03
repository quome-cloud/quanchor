//! Provider-independent internal representation (IR).
//!
//! Every provider adapter normalizes its native wire format to and from these
//! types so that a firewall chain is provider-independent. The firewall
//! evaluates the *prompt text* extracted from a request via
//! [`LlmRequest::prompt_text`].

use serde::{Deserialize, Serialize};

/// A message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn tag(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A single conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message { role, content: content.into() }
    }
}

/// A normalized tool/function definition advertised to the model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON-schema parameters, kept opaque.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// Sampling/generation parameters, normalized.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

/// A provider-independent LLM request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub params: GenParams,
    #[serde(default)]
    pub stream: bool,
}

impl LlmRequest {
    /// A minimal single-user-message request, handy for the CLI and tests.
    pub fn user(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        LlmRequest {
            model: model.into(),
            system: None,
            messages: vec![Message::new(Role::User, prompt)],
            tools: Vec::new(),
            params: GenParams::default(),
            stream: false,
        }
    }

    /// The flattened prompt text the firewall evaluates: the system prompt (if
    /// any) followed by each message, tagged with its role. This is what
    /// detectors see, making evaluation provider- and shape-independent.
    pub fn prompt_text(&self) -> String {
        let mut out = String::new();
        if let Some(sys) = &self.system {
            out.push_str("[system] ");
            out.push_str(sys);
            out.push('\n');
        }
        for m in &self.messages {
            out.push('[');
            out.push_str(m.role.tag());
            out.push_str("] ");
            out.push_str(&m.content);
            out.push('\n');
        }
        out.trim_end().to_string()
    }

    /// Just the most recent user turn, used by detectors that focus on the
    /// active request rather than the full transcript.
    pub fn latest_user(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
    }
}

/// Token usage and an estimated cost for a downstream call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub cost_usd: f64,
}

impl Usage {
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// A provider-independent LLM response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub model: String,
    pub content: String,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub finish_reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_text_includes_roles_and_system() {
        let mut req = LlmRequest::user("m", "write a tagline");
        req.system = Some("You are helpful".into());
        let text = req.prompt_text();
        assert!(text.contains("[system] You are helpful"));
        assert!(text.contains("[user] write a tagline"));
    }

    #[test]
    fn latest_user_picks_last_user_turn() {
        let req = LlmRequest {
            model: "m".into(),
            system: None,
            messages: vec![
                Message::new(Role::User, "first"),
                Message::new(Role::Assistant, "reply"),
                Message::new(Role::User, "second"),
            ],
            tools: vec![],
            params: GenParams::default(),
            stream: false,
        };
        assert_eq!(req.latest_user(), Some("second"));
    }
}
