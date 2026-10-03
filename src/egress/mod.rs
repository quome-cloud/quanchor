//! PHI egress control (paper 006). Value-set provenance taint-tracking plus content-DLP baselines
//! over outbound tool calls. This module scores detectors; policy enforcement (005 integration) and
//! anomaly/k-anon land in later plans.

pub mod anomaly;
pub mod content;
pub mod guard;
pub mod kanon;
pub mod normalize;
pub mod taint;

/// One PHI-egress finding: which tagged source value leaked, where, and via what transform.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct EgressFinding {
    pub prov_id: String,
    pub label: String,
    pub arg_path: String,
    pub via: String, // "verbatim" | "base64" | "hex" | "rot13" | "casefold" | "content"
}

/// A detector scores one outbound call (its args, flattened to strings) for PHI egress.
pub trait Detector {
    /// Detector name for reporting.
    fn name(&self) -> &str;
    /// Return findings for this outbound call's arg strings.
    fn scan(&self, arg_strings: &[(String, String)]) -> Vec<EgressFinding>; // (arg_path, value)
}
