//! Paper 004 — AI supply-chain attestation. At startup the gateway enumerates
//! every component it loads into a signed CycloneDX AIBOM, verifies digests and
//! attestations, matches against a vulnerability feed, and fail-closed gates the
//! service. The verified AIBOM digest is stamped into the 003 audit chain.

pub mod aibom;
pub mod attest;
pub mod gate;
pub mod vuln;

pub use gate::{AdmissionReport, Enforce};

use crate::audit::AuditSink;
use crate::Result;
use std::path::Path;

/// Admission configuration (a section of the main Config).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AdmissionCfg {
    pub enforce: Enforce,
    /// Pinned OSV snapshot path ("" disables vulnerability matching).
    pub osv_snapshot: String,
    /// Path to Cargo.lock for library enumeration ("" disables).
    pub cargo_lock: String,
    /// Optional ONNX detector model path ("" if none).
    pub onnx_path: String,
}

impl Default for AdmissionCfg {
    fn default() -> Self {
        AdmissionCfg {
            enforce: Enforce::Warn, // dev default; production enclave sets `block`
            osv_snapshot: String::new(),
            cargo_lock: "Cargo.lock".into(),
            onnx_path: String::new(),
        }
    }
}

/// Run the admission check at startup. Enumerates components, verifies digests,
/// matches vulnerabilities, decides, and stamps the outcome into the audit chain.
/// Returns the report; the caller fails closed when `!report.admitted`.
pub fn run_admission(
    cfg: &AdmissionCfg,
    rules_dir: &Path,
    chains_dir: &Path,
    config_path: Option<&Path>,
    provider_models: &[String],
    audit: &AuditSink,
) -> Result<AdmissionReport> {
    if cfg.enforce == Enforce::Off {
        let empty = aibom::Aibom::default();
        let report = gate::decide(&empty, &[], &[], Enforce::Off);
        return Ok(report);
    }

    let onnx = (!cfg.onnx_path.is_empty()).then(|| std::path::PathBuf::from(&cfg.onnx_path));
    let cargo = (!cfg.cargo_lock.is_empty()).then(|| std::path::PathBuf::from(&cfg.cargo_lock));
    let aibom = aibom::Aibom::enumerate(
        rules_dir,
        chains_dir,
        config_path,
        onnx.as_deref(),
        cargo.as_deref(),
        provider_models,
    );

    let tampered: Vec<String> = attest::tampered_components(&aibom)
        .into_iter()
        .map(|c| c.name)
        .collect();

    let vulnerable = if cfg.osv_snapshot.is_empty() {
        Vec::new()
    } else {
        use vuln::VulnFeed;
        vuln::OsvSnapshot::load(Path::new(&cfg.osv_snapshot))?.matches(&aibom)
    };

    let report = gate::decide(&aibom, &tampered, &vulnerable, cfg.enforce);
    gate::stamp(&report, audit)?;
    Ok(report)
}
