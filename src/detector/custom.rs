//! Custom detector node: shells out to a user-supplied script.
//!
//! The script receives a JSON object on stdin (`{"prompt": ..., "scope": ...}`)
//! and must print a JSON verdict on stdout
//! (`{"verdict": "allow"|"block"|"abstain", "confidence": 0.0-1.0,
//! "rationale": "..."}`). A non-zero exit is treated as an ERROR.

use super::{elapsed_ms, now, DetectCtx, Detector};
use crate::verdict::{NodeVerdict, Verdict};
use async_trait::async_trait;
use serde::Deserialize;

pub struct CustomDetector {
    command: String,
    config_hash: String,
}

impl CustomDetector {
    pub fn new(command: String, config_hash: String) -> Self {
        CustomDetector { command, config_hash }
    }
}

#[derive(Deserialize)]
struct CustomVerdict {
    verdict: String,
    #[serde(default = "half")]
    confidence: f64,
    #[serde(default)]
    rationale: String,
}

fn half() -> f64 {
    0.5
}

#[async_trait]
impl Detector for CustomDetector {
    fn kind(&self) -> &'static str {
        "custom"
    }
    fn version(&self) -> String {
        format!("custom/{}", self.command)
    }
    fn config_hash(&self) -> &str {
        &self.config_hash
    }
    fn is_expensive(&self) -> bool {
        true
    }

    async fn evaluate(&self, ctx: &DetectCtx<'_>) -> NodeVerdict {
        let start = now();
        let input = serde_json::json!({ "prompt": ctx.prompt, "scope": ctx.scope }).to_string();

        // Split the command on whitespace; first token is the program.
        let mut parts = self.command.split_whitespace();
        let program = match parts.next() {
            Some(p) => p,
            None => {
                return NodeVerdict::new(
                    self.kind(),
                    self.version(),
                    Verdict::Error,
                    0.0,
                    elapsed_ms(start),
                    "empty custom command",
                )
            }
        };
        let args: Vec<&str> = parts.collect();

        let result = tokio::process::Command::new(program)
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let mut child = match result {
            Ok(c) => c,
            Err(e) => {
                return NodeVerdict::new(
                    self.kind(),
                    self.version(),
                    Verdict::Error,
                    0.0,
                    elapsed_ms(start),
                    format!("failed to spawn custom command: {e}"),
                )
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(input.as_bytes()).await;
        }
        let output = match child.wait_with_output().await {
            Ok(o) => o,
            Err(e) => {
                return NodeVerdict::new(
                    self.kind(),
                    self.version(),
                    Verdict::Error,
                    0.0,
                    elapsed_ms(start),
                    format!("custom command error: {e}"),
                )
            }
        };
        if !output.status.success() {
            return NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Error,
                0.0,
                elapsed_ms(start),
                format!("custom command exited with {}", output.status),
            );
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        match serde_json::from_str::<CustomVerdict>(stdout.trim()) {
            Ok(cv) => {
                let verdict = match cv.verdict.to_ascii_lowercase().as_str() {
                    "allow" => Verdict::Allow,
                    "block" => Verdict::Block,
                    _ => Verdict::Abstain,
                };
                NodeVerdict::new(
                    self.kind(),
                    self.version(),
                    verdict,
                    cv.confidence,
                    elapsed_ms(start),
                    if cv.rationale.is_empty() { "custom verdict".into() } else { cv.rationale },
                )
            }
            Err(e) => NodeVerdict::new(
                self.kind(),
                self.version(),
                Verdict::Error,
                0.0,
                elapsed_ms(start),
                format!("custom output not valid verdict JSON: {e}"),
            ),
        }
    }
}
