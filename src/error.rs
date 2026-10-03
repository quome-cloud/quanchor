//! Error types for QFIRE.

use thiserror::Error;

/// The crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced across the QFIRE engine, providers and CLI.
#[derive(Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("invalid regex: {0}")]
    Regex(#[from] regex::Error),

    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),

    /// A rule or chain referenced something that does not exist.
    #[error("reference error: {0}")]
    Reference(String),

    /// A rule, chain or node was malformed.
    #[error("config error: {0}")]
    Config(String),

    /// A provider profile is missing or misconfigured.
    #[error("provider error: {0}")]
    Provider(String),

    /// The expression-mode chain DAG failed to parse.
    #[error("expression error: {0}")]
    Expression(String),

    /// A downstream call or detector failed in a way that fail-closed policy
    /// should treat as a BLOCK.
    #[error("detector error: {0}")]
    Detector(String),

    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Error::Other(e.to_string())
    }
}
