//! Change-fingerprinter. A `Fingerprint` is the SHA-256 digest of each component
//! class of an agent's deployed configuration — weights, prompt, tools, data. The
//! gate computes a *live* fingerprint at admission and diffs it against the one
//! pinned in the passport; each differing component is a `ComponentDelta` the PCCP
//! evaluator (see [`super::pccp`]) then judges in- or out-of-envelope. Component
//! granularity is the unit of L2 scope-creep detection (H4: detect 100% of swaps).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The four component classes whose change is governed by the PCCP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Weights,
    Prompt,
    Tools,
    Data,
}

impl Component {
    pub const ALL: [Component; 4] =
        [Component::Weights, Component::Prompt, Component::Tools, Component::Data];
    pub fn name(self) -> &'static str {
        match self {
            Component::Weights => "weights",
            Component::Prompt => "prompt",
            Component::Tools => "tools",
            Component::Data => "data",
        }
    }
}

/// SHA-256 (hex) of each component class of a deployed agent config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub weights: String,
    pub prompt: String,
    pub tools: String,
    pub data: String,
}

/// One component whose digest changed between the deployed and live config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentDelta {
    pub component: Component,
    pub from: String,
    pub to: String,
    /// Optional magnitude hint in [0,1] (e.g. fraction of weights changed) supplied
    /// by the producer of the change; `None` means "magnitude unknown".
    #[serde(default)]
    pub magnitude: Option<f64>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

impl Fingerprint {
    /// Hash each component's bytes into a fingerprint.
    pub fn of(weights: &[u8], prompt: &[u8], tools: &[u8], data: &[u8]) -> Self {
        Fingerprint {
            weights: sha256_hex(weights),
            prompt: sha256_hex(prompt),
            tools: sha256_hex(tools),
            data: sha256_hex(data),
        }
    }

    fn digest(&self, c: Component) -> &str {
        match c {
            Component::Weights => &self.weights,
            Component::Prompt => &self.prompt,
            Component::Tools => &self.tools,
            Component::Data => &self.data,
        }
    }
}

/// Diff a deployed fingerprint against a live one; one `ComponentDelta` per class
/// whose digest changed. Magnitudes are unknown here (the caller may attach them).
pub fn diff(deployed: &Fingerprint, live: &Fingerprint) -> Vec<ComponentDelta> {
    let mut out = Vec::new();
    for c in Component::ALL {
        let (from, to) = (deployed.digest(c), live.digest(c));
        if from != to {
            out.push(ComponentDelta {
                component: c,
                from: from.to_string(),
                to: to.to_string(),
                magnitude: None,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_of_bytes_is_sha256() {
        let fp = Fingerprint::of(b"w", b"p", b"t", b"d");
        // Known SHA-256 of "w".
        assert_eq!(
            fp.weights,
            "50e721e49c013f00c62cf59f2163542a9d8df02464efeb615d31051b0fddc326"
        );
        assert_eq!(fp.weights.len(), 64);
    }

    #[test]
    fn diff_detects_single_component_swap() {
        let a = Fingerprint::of(b"w", b"p", b"t", b"d");
        let b = Fingerprint::of(b"w", b"PROMPT-CHANGED", b"t", b"d");
        let d = diff(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].component, Component::Prompt);
        assert_ne!(d[0].from, d[0].to);
    }

    #[test]
    fn diff_detects_multi_swap() {
        let a = Fingerprint::of(b"w", b"p", b"t", b"d");
        let b = Fingerprint::of(b"W2", b"p", b"T2", b"d");
        let d = diff(&a, &b);
        let comps: Vec<_> = d.iter().map(|x| x.component).collect();
        assert_eq!(d.len(), 2);
        assert!(comps.contains(&Component::Weights));
        assert!(comps.contains(&Component::Tools));
    }

    #[test]
    fn diff_identical_is_empty() {
        let a = Fingerprint::of(b"w", b"p", b"t", b"d");
        assert!(diff(&a, &a.clone()).is_empty());
    }
}
