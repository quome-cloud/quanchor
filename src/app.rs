//! High-level application wiring shared by the CLI and proxy.
//!
//! [`App`] loads configuration, builds the provider registry, indexes the rule
//! library and chain definitions, constructs the evaluation engine, and opens the
//! audit log. It exposes the three core operations — `check` (evaluate only),
//! `run` (evaluate and, on ALLOW, execute the downstream call) and `explain`
//! (dry-run, no audit, no provider) — so every surface shares one engine.

use crate::audit::{AuditLog, AuditRecord, AuditSink};
use crate::chain::{Chain, ChainMode, FailPolicy};
use crate::config::Config;
use crate::engine::{Decision, Engine};
use crate::ir::{LlmRequest, LlmResponse};
use crate::provider::ProviderRegistry;
use crate::rule::CompiledRule;
use crate::ruleset::RuleSet;
use crate::verdict::Verdict;
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use walkdir::WalkDir;

/// The assembled QFIRE application.
pub struct App {
    pub config: Config,
    pub rules: RuleSet,
    pub chains: HashMap<String, Chain>,
    pub engine: Engine,
    pub audit: AuditSink,
}

impl App {
    /// Load the application from configuration (searching standard paths).
    pub fn load(config_path: Option<&Path>) -> Result<App> {
        let config = Config::load(config_path)?;
        Self::from_config(config)
    }

    /// Build the application from an explicit config.
    pub fn from_config(config: Config) -> Result<App> {
        let providers = Arc::new(ProviderRegistry::from_profiles(&config.providers)?);
        let rules = RuleSet::load_dir(Path::new(&config.rules_dir))?;
        let chains = load_chains(Path::new(&config.chains_dir))?;
        let engine = Engine::new(providers);
        let audit = match config.audit.mode {
            crate::audit::store::Mode::Plain => {
                AuditSink::Plain(AuditLog::open(&config.audit_path))
            }
            mode => {
                let signer = crate::audit::sign::AuditSigner::load_or_generate(
                    std::path::Path::new(&config.audit.key_path),
                )?;
                let anchor = (!config.audit.anchors_path.is_empty()
                    && config.audit.anchor_every > 0)
                    .then(|| {
                        (
                            Box::new(crate::audit::anchor_sink::FileAnchorSink::new(
                                &config.audit.anchors_path,
                            ))
                                as Box<dyn crate::audit::anchor_sink::AnchorSink>,
                            config.audit.anchor_every,
                        )
                    });
                AuditSink::Chained(crate::audit::store::TamperEvidentLog::open(
                    crate::audit::store::StoreCfg {
                        path: std::path::PathBuf::from(&config.audit_path),
                        mode,
                        signer: Some(signer),
                        anchor,
                        batch: config.audit.batch,
                        fail_open: config.audit.fail_open,
                    },
                )?)
            }
        };
        // Paper 004: supply-chain admission gate. Fail-closed — an unverified or
        // known-vulnerable component blocks startup under enforce=block.
        let provider_models: Vec<String> = config.providers.iter()
            .filter_map(|p| p.model.clone()).collect();
        let report = crate::admission::run_admission(
            &config.admission,
            Path::new(&config.rules_dir),
            Path::new(&config.chains_dir),
            // config path not threaded here: qfire.toml is absent from the runtime
            // AIBOM (it raises the provenance gap); use `qfire aibom generate --config`
            // to include it in an offline AIBOM.
            None,
            &provider_models,
            &audit,
        )?;
        if !report.admitted {
            return Err(crate::error::Error::Config(format!(
                "supply-chain admission denied: {} tampered, {} vulnerable component(s); \
                 set [admission] enforce=\"warn\" to override (dev only)",
                report.tampered.len(), report.vulnerable.len()
            )));
        }
        if report.enforce == crate::admission::Enforce::Warn
            && (!report.tampered.is_empty() || !report.vulnerable.is_empty()) {
            tracing::warn!(
                "supply-chain admission (warn): {} tampered, {} vulnerable; proceeding",
                report.tampered.len(), report.vulnerable.len());
        }
        Ok(App { config, rules, chains, engine, audit })
    }

    /// Disable the audit log (e.g. for `explain` dry-runs).
    pub fn without_audit(mut self) -> Self {
        self.audit = AuditSink::disabled();
        self
    }

    /// Resolve a chain by id. If no chain matches but a *rule* does, synthesize a
    /// single-rule ordered chain with default-deny — convenient for ad-hoc checks.
    pub fn resolve_chain(&self, name: &str) -> Result<Chain> {
        if let Some(c) = self.chains.get(name) {
            return Ok(c.clone());
        }
        if self.rules.contains(name) {
            return Ok(Chain {
                id: format!("adhoc:{name}"),
                description: Some(format!("synthesized single-rule chain for '{name}'")),
                mode: ChainMode::Ordered,
                fail_policy: FailPolicy::FailClosed,
                rules: vec![name.to_string()],
                // A single-rule check blocks only when the rule explicitly
                // BLOCKs; an ABSTAIN (e.g. a guard on a clean prompt) passes.
                default: Verdict::Allow,
                expression: None,
                groups: HashMap::new(),
                provider: None,
                normalize: false,
                normalize_mode: None,
                version: "adhoc".into(),
            });
        }
        Err(Error::Reference(format!("no chain or rule named '{name}'")))
    }

    /// Compile the rules a chain references.
    pub fn compile_for(&self, chain: &Chain) -> Result<HashMap<String, CompiledRule>> {
        let ids = chain.referenced_rules()?;
        self.rules.compile_subset(&ids)
    }

    /// Evaluate a request against a chain (no downstream call). Audited.
    pub async fn check(&self, chain_name: &str, request: &LlmRequest) -> Result<Decision> {
        let chain = self.resolve_chain(chain_name)?;
        let compiled = self.compile_for(&chain)?;
        let decision = self.engine.evaluate(&chain, &compiled, request).await?;
        self.audit.append(&AuditRecord::from_decision("check", &decision))?;
        Ok(decision)
    }

    /// Evaluate and, on ALLOW, execute the downstream call. Audited.
    pub async fn run(
        &self,
        chain_name: &str,
        request: &LlmRequest,
        provider_override: Option<&str>,
    ) -> Result<(Decision, Option<LlmResponse>)> {
        let chain = self.resolve_chain(chain_name)?;
        let compiled = self.compile_for(&chain)?;
        let decision = self.engine.evaluate(&chain, &compiled, request).await?;

        if !decision.allowed() {
            self.audit.append(&AuditRecord::from_decision("run.block", &decision))?;
            return Ok((decision, None));
        }

        // Resolve the downstream provider: explicit override, chain's provider,
        // else the registry default.
        let provider_name = provider_override
            .map(|s| s.to_string())
            .or_else(|| chain.provider.clone());
        let provider = match &provider_name {
            Some(n) => self.engine.providers().get(n)?,
            None => self.engine.providers().default()?,
        };
        let response = provider.complete(request).await?;
        let record = AuditRecord::from_decision("run.allow", &decision)
            .with_downstream(provider.name(), &response.model, response.usage.clone());
        self.audit.append(&record)?;
        Ok((decision, Some(response)))
    }

    /// Dry-run a chain without auditing or contacting any provider for the
    /// downstream call (the judge node may still call a provider for evaluation).
    pub async fn explain(&self, chain_name: &str, request: &LlmRequest) -> Result<Decision> {
        let chain = self.resolve_chain(chain_name)?;
        let compiled = self.compile_for(&chain)?;
        self.engine.evaluate(&chain, &compiled, request).await
    }
}

/// Load every chain definition under `dir` (recursively). Missing dir is OK
/// (returns an empty map) so QFIRE works before any chains are authored.
pub fn load_chains(dir: &Path) -> Result<HashMap<String, Chain>> {
    let mut out = HashMap::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        let is_yaml = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e == "yaml" || e == "yml")
            .unwrap_or(false);
        if !is_yaml || !path.is_file() {
            continue;
        }
        let chain = Chain::from_path(path)?;
        if let Some(existing) = out.insert(chain.id.clone(), chain) {
            return Err(Error::Config(format!("duplicate chain id: {}", existing.id)));
        }
    }
    Ok(out)
}
