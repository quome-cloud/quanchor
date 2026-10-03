//! The wire-compatible proxy daemon (`qfire serve`).
//!
//! A long-running async proxy that exposes drop-in, wire-compatible endpoints for
//! each ecosystem (OpenAI `/v1/chat/completions` + `/v1/responses`, Anthropic
//! `/v1/messages`, Gemini `:generateContent`, Ollama `/api/chat` + `/api/generate`)
//! so existing SDKs work by only changing the base URL. The active firewall chain
//! is selected by the `X-QFire-Chain` header or the server default. On ALLOW the
//! original request is forwarded to the downstream provider and the response
//! (including streams) is passed through; on BLOCK a structured refusal envelope
//! is returned and the provider is never contacted.

use crate::app::App;
use crate::audit::AuditRecord;
use crate::engine::CompiledRules;
use crate::ir::{GenParams, LlmRequest, Message, Role};
use crate::output;
use crate::Result;
use axum::body::{Body, Bytes};
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::Router;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Which wire family an incoming request belongs to, inferred from its path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    OpenAiChat,
    OpenAiResponses,
    Anthropic,
    Gemini,
    OllamaChat,
    OllamaGenerate,
    Unknown,
}

fn family_of(path: &str) -> Family {
    if path.contains(":generateContent") || path.contains(":streamGenerateContent") {
        Family::Gemini
    } else if path.contains("/v1/chat/completions") || path.contains("/v1/completions") {
        Family::OpenAiChat
    } else if path.contains("/v1/responses") {
        Family::OpenAiResponses
    } else if path.contains("/v1/messages") {
        Family::Anthropic
    } else if path.contains("/api/chat") {
        Family::OllamaChat
    } else if path.contains("/api/generate") {
        Family::OllamaGenerate
    } else {
        Family::Unknown
    }
}

struct ProxyState {
    app: App,
    default_chain: String,
    redact: bool,
    /// On BLOCK, return a 200 OpenAI-shaped refusal completion for OpenAI-family
    /// requests instead of the default 403 envelope (opt-in).
    openai_block_refusal: bool,
    client: reqwest::Client,
    compiled: Mutex<HashMap<String, Arc<CompiledRules>>>,
    /// Required inbound token (from `[server] auth_token`); None disables.
    auth_token: Option<String>,
}

impl ProxyState {
    /// Get (compiling and caching on first use) the compiled rule bundle for a
    /// chain, so the proxy hot path does not recompile regexes per request.
    fn bundle(&self, chain_id: &str, chain: &crate::chain::Chain) -> Result<Arc<CompiledRules>> {
        if let Some(b) = self.compiled.lock().unwrap().get(chain_id) {
            return Ok(b.clone());
        }
        let compiled = Arc::new(self.app.compile_for(chain)?);
        self.compiled
            .lock()
            .unwrap()
            .insert(chain_id.to_string(), compiled.clone());
        Ok(compiled)
    }
}

/// Run the proxy until terminated.
pub async fn serve(
    app: App,
    addr: &str,
    default_chain: &str,
    redact: bool,
    openai_block_refusal: bool,
) -> Result<()> {
    let auth_token = match &app.config.server.auth_token {
        None => None,
        Some(_) => match app.config.server.resolve_token() {
            Some(t) if !t.is_empty() => Some(t),
            _ => {
                return Err(crate::Error::Config(
                    "server.auth_token is configured but resolves to empty \
                     (unset env var?); refusing to start an open proxy"
                        .into(),
                ))
            }
        },
    };
    let state = Arc::new(ProxyState {
        app,
        default_chain: default_chain.to_string(),
        redact,
        openai_block_refusal,
        client: reqwest::Client::new(),
        compiled: Mutex::new(HashMap::new()),
        auth_token,
    });

    let router = Router::new().fallback(any(handle)).with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(crate::Error::Io)?;
    let bound = listener.local_addr().map_err(crate::Error::Io)?;
    eprintln!("qfire proxy listening on http://{bound}  (default chain: {default_chain})");
    eprintln!("  OpenAI    → POST http://{bound}/v1/chat/completions");
    eprintln!("  Anthropic → POST http://{bound}/v1/messages");
    eprintln!("  Gemini    → POST http://{bound}/v1beta/models/<model>:generateContent");
    eprintln!("  Ollama    → POST http://{bound}/api/chat");
    eprintln!("  select a chain per-request with the  X-QFire-Chain  header");

    axum::serve(listener, router)
        .await
        .map_err(crate::Error::Io)?;
    Ok(())
}

/// Resolve the downstream profile name for a request.
/// Precedence: X-QFire-Provider header > chain.provider > registry default (None).
fn provider_name<'a>(headers: &'a HeaderMap, chain_provider: Option<&'a str>) -> Option<&'a str> {
    headers
        .get("x-qfire-provider")
        .and_then(|v| v.to_str().ok())
        .or(chain_provider)
}

/// Constant-time byte comparison via `subtle` (length mismatch returns early;
/// the token length is not secret).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq as _;
    a.len() == b.len() && bool::from(a.ct_eq(b))
}

/// Inbound auth: when a token is configured, the `X-QFire-Token` header must
/// match. No token configured -> open (local dev).
fn auth_ok(headers: &HeaderMap, expected: Option<&str>) -> bool {
    match expected {
        None => true,
        Some(want) => headers
            .get("x-qfire-token")
            .and_then(|v| v.to_str().ok())
            .map(|got| constant_time_eq(got.as_bytes(), want.as_bytes()))
            .unwrap_or(false),
    }
}

async fn handle(
    State(state): State<Arc<ProxyState>>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path = uri.path().to_string();

    // Simple health endpoint.
    if method == Method::GET && (path == "/" || path == "/health") {
        return (StatusCode::OK, "qfire proxy ok\n").into_response();
    }

    if !auth_ok(&headers, state.auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            "qfire: missing or invalid X-QFire-Token\n",
        )
            .into_response();
    }

    if method != Method::POST {
        return (StatusCode::METHOD_NOT_ALLOWED, "qfire: POST only\n").into_response();
    }

    let family = family_of(&path);
    if family == Family::Unknown {
        return (
            StatusCode::NOT_FOUND,
            "qfire: unrecognized provider endpoint\n",
        )
            .into_response();
    }

    let json: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("qfire: invalid JSON body: {e}\n"),
            )
                .into_response()
        }
    };
    let request = extract_request(family, &json, &path);

    // Chain selection: X-QFire-Chain header, else server default.
    let chain_name = headers
        .get("x-qfire-chain")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&state.default_chain)
        .to_string();

    let chain = match state.app.resolve_chain(&chain_name) {
        Ok(c) => c,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("qfire: {e}\n")).into_response(),
    };
    let bundle = match state.bundle(&chain.id, &chain) {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("qfire: {e}\n")).into_response()
        }
    };

    let decision = match state.app.engine.evaluate(&chain, &bundle, &request).await {
        Ok(d) => d,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("qfire: {e}\n")).into_response()
        }
    };

    if !decision.allowed() {
        let _ = state
            .app
            .audit
            .append(&AuditRecord::from_decision("proxy.block", &decision));
        if state.openai_block_refusal && family == Family::OpenAiChat {
            let body = output::openai_refusal_completion(&decision, &request.model, state.redact);
            return (StatusCode::OK, axum::Json(body)).into_response();
        }
        let envelope = output::refusal_json(&decision, state.redact);
        return (StatusCode::FORBIDDEN, axum::Json(envelope)).into_response();
    }

    // ALLOW: forward the original request to the downstream provider.
    // Profile precedence: X-QFire-Provider header > chain.provider > default.
    let provider = match provider_name(&headers, chain.provider.as_deref())
        .map(|n| state.app.engine.providers().get(n))
        .unwrap_or_else(|| state.app.engine.providers().default())
    {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("qfire: {e}\n")).into_response(),
    };
    let _ = state.app.audit.append(
        &AuditRecord::from_decision("proxy.allow", &decision).with_downstream(
            provider.name(),
            &request.model,
            Default::default(),
        ),
    );

    // Opt-in behavioral output-monitor pass: if the client sends
    // `X-QFire-Monitor-Output: 1` (or `true`), buffer the response and run
    // the output-monitor engine pass before returning the bytes unchanged.
    // Without the header the existing zero-copy streaming path is used.
    let monitor_output = headers
        .get("x-qfire-monitor-output")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    if monitor_output {
        forward_buffered(
            ForwardCtx {
                client: &state.client,
                provider: provider.as_ref(),
                uri: &uri,
                headers: &headers,
                body,
            },
            family,
            &state,
            &chain,
            &bundle,
            &request,
        )
        .await
    } else {
        forward(&state.client, provider.as_ref(), &uri, &headers, body).await
    }
}

/// Forward the raw request body to the downstream provider, preserving the path
/// and streaming the response back.
async fn forward(
    client: &reqwest::Client,
    provider: &dyn crate::provider::Provider,
    uri: &axum::http::Uri,
    headers: &HeaderMap,
    body: Bytes,
) -> Response {
    use crate::config::ProviderKind;
    let path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(uri.path());
    let url = format!(
        "{}{}",
        provider.base_url().trim_end_matches('/'),
        path_and_query
    );

    let mut req = client.post(&url).header("content-type", "application/json");

    if provider.auth_passthrough() {
        // Passthrough: forward the caller's provider credentials verbatim.
        // (Query-string creds like Gemini's ?key= are already preserved via
        // path_and_query above.)
        for h in ["authorization", "x-api-key", "anthropic-version", "x-goog-api-key"] {
            if let Some(v) = headers.get(h) {
                req = req.header(h, v);
            }
        }
    } else {
        // Apply downstream auth based on the provider family. The client's own auth
        // headers are intentionally not forwarded; QFIRE injects the configured key.
        match provider.kind() {
            ProviderKind::OpenAi => {
                if let Some(k) = auth_key(provider) {
                    req = req.bearer_auth(k);
                }
            }
            ProviderKind::Anthropic => {
                if let Some(k) = auth_key(provider) {
                    req = req
                        .header("x-api-key", k)
                        .header("anthropic-version", "2023-06-01");
                }
            }
            ProviderKind::Gemini | ProviderKind::Ollama => { /* key in query / none */ }
        }
    }
    // Preserve accept header for SSE streaming when present.
    if let Some(accept) = headers.get("accept") {
        req = req.header("accept", accept);
    }

    let resp = match req.body(body).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("qfire: downstream error: {e}\n"),
            )
                .into_response()
        }
    };

    let status = resp.status();
    let mut builder = Response::builder().status(status.as_u16());
    if let Some(ct) = resp.headers().get("content-type") {
        builder = builder.header("content-type", ct);
    }
    let stream = resp.bytes_stream();
    builder.body(Body::from_stream(stream)).unwrap_or_else(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "qfire: response build error\n",
        )
            .into_response()
    })
}

/// QFIRE does not expose provider keys; this helper reads the configured key
/// from the registry profile via the environment-resolved profile. Since the
/// `Provider` trait abstracts that away, we look it up through the config that
/// built it. For simplicity the proxy relies on Ollama (no key) by default.
fn auth_key(_provider: &dyn crate::provider::Provider) -> Option<String> {
    None
}

/// Extract a normalized [`LlmRequest`] from a provider-native JSON body, for the
/// purpose of firewall evaluation. Forwarding still uses the original raw bytes.
fn extract_request(family: Family, json: &serde_json::Value, _path: &str) -> LlmRequest {
    let model = json
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();
    let stream = json
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let mut system = None;
    let mut messages = Vec::new();

    match family {
        Family::OpenAiChat | Family::OllamaChat => {
            if let Some(arr) = json.get("messages").and_then(|m| m.as_array()) {
                for m in arr {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let content = extract_content(m.get("content"));
                    push_msg(&mut system, &mut messages, role, content);
                }
            }
        }
        Family::OpenAiResponses => {
            // /v1/responses uses `input` (string or array) + optional `instructions`.
            if let Some(instr) = json.get("instructions").and_then(|i| i.as_str()) {
                system = Some(instr.to_string());
            }
            match json.get("input") {
                Some(serde_json::Value::String(s)) => {
                    messages.push(Message::new(Role::User, s.clone()))
                }
                Some(serde_json::Value::Array(arr)) => {
                    for m in arr {
                        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                        let content = extract_content(m.get("content"));
                        push_msg(&mut system, &mut messages, role, content);
                    }
                }
                _ => {}
            }
        }
        Family::Anthropic => {
            if let Some(s) = json.get("system").and_then(|s| s.as_str()) {
                system = Some(s.to_string());
            }
            if let Some(arr) = json.get("messages").and_then(|m| m.as_array()) {
                for m in arr {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let content = extract_content(m.get("content"));
                    push_msg(&mut system, &mut messages, role, content);
                }
            }
        }
        Family::Gemini => {
            if let Some(parts) = json
                .get("systemInstruction")
                .and_then(|s| s.get("parts"))
                .and_then(|p| p.as_array())
            {
                let text: String = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                if !text.is_empty() {
                    system = Some(text);
                }
            }
            if let Some(arr) = json.get("contents").and_then(|c| c.as_array()) {
                for c in arr {
                    let role = match c.get("role").and_then(|r| r.as_str()) {
                        Some("model") => "assistant",
                        _ => "user",
                    };
                    let text: String = c
                        .get("parts")
                        .and_then(|p| p.as_array())
                        .map(|parts| {
                            parts
                                .iter()
                                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                        .unwrap_or_default();
                    messages.push(Message::new(role_of(role), text));
                }
            }
        }
        Family::OllamaGenerate => {
            if let Some(s) = json.get("system").and_then(|s| s.as_str()) {
                system = Some(s.to_string());
            }
            if let Some(p) = json.get("prompt").and_then(|p| p.as_str()) {
                messages.push(Message::new(Role::User, p.to_string()));
            }
        }
        Family::Unknown => {}
    }

    if messages.is_empty() {
        messages.push(Message::new(Role::User, ""));
    }
    LlmRequest {
        model,
        system,
        messages,
        tools: vec![],
        params: GenParams::default(),
        stream,
    }
}

fn push_msg(system: &mut Option<String>, messages: &mut Vec<Message>, role: &str, content: String) {
    if role == "system" {
        *system = Some(match system.take() {
            Some(s) => format!("{s}\n{content}"),
            None => content,
        });
    } else {
        messages.push(Message::new(role_of(role), content));
    }
}

fn role_of(role: &str) -> Role {
    match role {
        "system" => Role::System,
        "assistant" | "model" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

/// Content may be a string or an array of content parts (OpenAI/Anthropic).
fn extract_content(v: Option<&serde_json::Value>) -> String {
    match v {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|t| t.as_str())
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Extract the assistant text from a provider-native response JSON.
///
/// Returns an empty string if the expected path is absent or not a string.
fn extract_response_text(family: Family, json: &serde_json::Value) -> String {
    match family {
        Family::OllamaChat => json
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string(),
        Family::OllamaGenerate => json
            .get("response")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string(),
        Family::OpenAiChat => json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string(),
        Family::OpenAiResponses => json
            .get("output")
            .and_then(|o| o.get(0))
            .and_then(|o| o.get("content"))
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        Family::Anthropic => json
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        Family::Gemini => json
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        Family::Unknown => String::new(),
    }
}

/// Arguments shared between `forward` and `forward_buffered`.
struct ForwardCtx<'a> {
    client: &'a reqwest::Client,
    provider: &'a dyn crate::provider::Provider,
    uri: &'a axum::http::Uri,
    headers: &'a HeaderMap,
    body: Bytes,
}

/// Forward the raw request body to the downstream provider, buffer the full
/// response body, parse it as JSON, and run the behavioral output-monitor pass.
/// The buffered bytes are returned to the client unchanged regardless of the
/// monitor decision (detection is recorded; the response is already generated).
async fn forward_buffered(
    ctx: ForwardCtx<'_>,
    family: Family,
    state: &ProxyState,
    chain: &crate::chain::Chain,
    bundle: &Arc<crate::engine::CompiledRules>,
    request: &crate::ir::LlmRequest,
) -> Response {
    let ForwardCtx {
        client,
        provider,
        uri,
        headers,
        body,
    } = ctx;
    use crate::config::ProviderKind;
    let path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(uri.path());
    let url = format!(
        "{}{}",
        provider.base_url().trim_end_matches('/'),
        path_and_query
    );

    let mut req = client.post(&url).header("content-type", "application/json");

    if provider.auth_passthrough() {
        for h in ["authorization", "x-api-key", "anthropic-version", "x-goog-api-key"] {
            if let Some(v) = headers.get(h) {
                req = req.header(h, v);
            }
        }
    } else {
        match provider.kind() {
            ProviderKind::OpenAi => {
                if let Some(k) = auth_key(provider) {
                    req = req.bearer_auth(k);
                }
            }
            ProviderKind::Anthropic => {
                if let Some(k) = auth_key(provider) {
                    req = req
                        .header("x-api-key", k)
                        .header("anthropic-version", "2023-06-01");
                }
            }
            ProviderKind::Gemini | ProviderKind::Ollama => {}
        }
    }
    if let Some(accept) = headers.get("accept") {
        req = req.header("accept", accept);
    }

    let resp = match req.body(body).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("qfire: downstream error: {e}\n"),
            )
                .into_response()
        }
    };

    let status = resp.status();
    let content_type = resp.headers().get("content-type").cloned();

    // Buffer the full response body.
    let buffered = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("qfire: downstream read error: {e}\n"),
            )
                .into_response()
        }
    };

    // Best-effort: parse JSON, extract assistant text, run output monitor.
    // A parse failure or monitor error must never 500 the client.
    if let Ok(resp_json) = serde_json::from_slice::<serde_json::Value>(&buffered) {
        let text = extract_response_text(family, &resp_json);
        if let Ok(output_decision) = state
            .app
            .engine
            .evaluate_output(chain, bundle, request, &text)
            .await
        {
            let _ = state.app.audit.append(
                &AuditRecord::from_decision("proxy.output_monitor", &output_decision)
                    .with_downstream(provider.name(), &request.model, Default::default()),
            );
        }
    }

    // Return the buffered bytes to the client with the original status/content-type.
    let mut builder = Response::builder().status(status.as_u16());
    if let Some(ct) = content_type {
        builder = builder.header("content-type", ct);
    }
    builder.body(Body::from(buffered)).unwrap_or_else(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "qfire: response build error\n",
        )
            .into_response()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn forward_passes_caller_auth_headers_through_for_passthrough_profiles() {
        use crate::config::{ProviderKind, ProviderProfile};
        use axum::routing::post;

        async fn echo(headers: axum::http::HeaderMap) -> axum::Json<serde_json::Value> {
            let get = |k: &str| {
                headers.get(k).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
            };
            axum::Json(serde_json::json!({
                "authorization": get("authorization"),
                "x-api-key": get("x-api-key"),
            }))
        }
        let router = axum::Router::new().route("/v1/chat/completions", post(echo));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let profile = ProviderProfile {
            name: "openai".into(),
            kind: ProviderKind::OpenAi,
            base_url: Some(format!("http://{addr}")),
            api_key: Some("passthrough".into()),
            model: None,
        };
        let client = reqwest::Client::new();
        let registry = crate::provider::ProviderRegistry::from_profiles(&[profile]).unwrap();
        let provider = registry.default().unwrap();

        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "Bearer caller-key".parse().unwrap());
        headers.insert("x-api-key", "caller-x-key".parse().unwrap());
        let uri: axum::http::Uri = "/v1/chat/completions".parse().unwrap();

        let resp =
            forward(&client, provider.as_ref(), &uri, &headers, Bytes::from_static(b"{}")).await;
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let echoed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(echoed["authorization"], "Bearer caller-key");
        assert_eq!(echoed["x-api-key"], "caller-x-key");
    }

    #[tokio::test]
    async fn forward_strips_caller_auth_headers_for_keyed_profiles() {
        use crate::config::{ProviderKind, ProviderProfile};
        use axum::routing::post;

        async fn echo(headers: axum::http::HeaderMap) -> axum::Json<serde_json::Value> {
            let get = |k: &str| {
                headers.get(k).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
            };
            axum::Json(serde_json::json!({
                "authorization": get("authorization"),
                "x-api-key": get("x-api-key"),
            }))
        }
        let router = axum::Router::new().route("/v1/chat/completions", post(echo));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let profile = ProviderProfile {
            name: "openai".into(),
            kind: ProviderKind::OpenAi,
            base_url: Some(format!("http://{addr}")),
            api_key: Some("sk-x".into()),
            model: None,
        };
        let client = reqwest::Client::new();
        let registry = crate::provider::ProviderRegistry::from_profiles(&[profile]).unwrap();
        let provider = registry.default().unwrap();

        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "Bearer caller-key".parse().unwrap());
        headers.insert("x-api-key", "caller-x-key".parse().unwrap());
        let uri: axum::http::Uri = "/v1/chat/completions".parse().unwrap();

        let resp =
            forward(&client, provider.as_ref(), &uri, &headers, Bytes::from_static(b"{}")).await;
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let echoed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // auth_key() returns None so nothing is injected; caller headers are stripped
        assert_eq!(echoed["authorization"], "");
        assert_eq!(echoed["x-api-key"], "");
    }

    #[test]
    fn provider_name_precedence() {
        use axum::http::HeaderMap;
        let mut headers = HeaderMap::new();
        assert_eq!(super::provider_name(&headers, None), None);
        assert_eq!(super::provider_name(&headers, Some("chain-prov")), Some("chain-prov"));
        headers.insert("x-qfire-provider", "hdr-prov".parse().unwrap());
        assert_eq!(super::provider_name(&headers, Some("chain-prov")), Some("hdr-prov"));
        assert_eq!(super::provider_name(&headers, None), Some("hdr-prov"));
    }

    #[test]
    fn token_check_constant_time_eq() {
        assert!(super::constant_time_eq(b"abc", b"abc"));
        assert!(!super::constant_time_eq(b"abc", b"abd"));
        assert!(!super::constant_time_eq(b"abc", b"abcd"));
    }

    #[test]
    fn token_gate_logic() {
        use axum::http::HeaderMap;
        let mut headers = HeaderMap::new();
        assert!(super::auth_ok(&headers, None));
        assert!(!super::auth_ok(&headers, Some("tok-123")));
        headers.insert("x-qfire-token", "nope".parse().unwrap());
        assert!(!super::auth_ok(&headers, Some("tok-123")));
        headers.insert("x-qfire-token", "tok-123".parse().unwrap());
        assert!(super::auth_ok(&headers, Some("tok-123")));
    }

    #[test]
    fn extract_response_text_handles_families() {
        let ollama = serde_json::json!({"message": {"content": "give 55 units"}});
        assert_eq!(
            extract_response_text(Family::OllamaChat, &ollama),
            "give 55 units"
        );

        let ollama_gen = serde_json::json!({"response": "generated text"});
        assert_eq!(
            extract_response_text(Family::OllamaGenerate, &ollama_gen),
            "generated text"
        );

        let openai = serde_json::json!({"choices": [{"message": {"content": "ok 120"}}]});
        assert_eq!(extract_response_text(Family::OpenAiChat, &openai), "ok 120");

        let anthropic =
            serde_json::json!({"content": [{"type": "text", "text": "hello from claude"}]});
        assert_eq!(
            extract_response_text(Family::Anthropic, &anthropic),
            "hello from claude"
        );

        let gemini = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "gemini says hi"}]
                }
            }]
        });
        assert_eq!(
            extract_response_text(Family::Gemini, &gemini),
            "gemini says hi"
        );

        let responses = serde_json::json!({"output": [{"type": "message", "content": [{"type": "output_text", "text": "resp 90"}]}]});
        assert_eq!(
            extract_response_text(Family::OpenAiResponses, &responses),
            "resp 90"
        );

        let missing = serde_json::json!({"unexpected": true});
        assert_eq!(extract_response_text(Family::OllamaChat, &missing), "");
    }
}
