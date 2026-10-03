//! Value-set provenance taint engine (paper 006). Tag PHI values seen from trusted inbound tool
//! results; on outbound calls, match any tagged value against the decoded/normalized arg strings —
//! catching verbatim, encoded, and split leaks because we match the ORIGIN value, not appearance.

use super::normalize::candidates;
use super::{Detector, EgressFinding};

#[derive(Default)]
pub struct TaintStore {
    /// (value, prov_id, label) tagged from trusted sources this session.
    tags: Vec<(String, String, String)>,
}

impl TaintStore {
    pub fn new() -> Self {
        TaintStore::default()
    }
    /// Tag a PHI value seen from a trusted inbound result.
    pub fn tag(&mut self, value: &str, prov_id: &str, label: &str) {
        self.tags.push((value.to_string(), prov_id.to_string(), label.to_string()));
    }

    /// (value, label) pairs of everything tagged this session — for redaction.
    pub fn tagged_values(&self) -> Vec<(String, String)> {
        self.tags.iter().map(|(v, _p, l)| (v.clone(), l.clone())).collect()
    }

    /// Scan one outbound call's (arg_path, value) strings for any tagged value, trying decoded
    /// candidates so encoded leaks still match. Split (X3) is caught by concatenating arg values.
    pub fn scan_call(&self, arg_strings: &[(String, String)]) -> Vec<EgressFinding> {
        let mut hits = Vec::new();
        // Per-arg candidates (verbatim + decoded). Also decode each whitespace-separated token so
        // an encoded leak embedded in surrounding text (e.g. "ref <base64>") is isolated/decoded.
        for (path, val) in arg_strings {
            for cand in candidates(val) {
                self.match_into(&cand, path, &mut hits);
            }
            for tok in val.split_whitespace() {
                for cand in candidates(tok) {
                    self.match_into(&cand, path, &mut hits);
                }
            }
        }
        // X3 split: the concatenation of all arg values (order-preserving) and its casefold form.
        let joined: String = arg_strings.iter().map(|(_, v)| v.as_str()).collect();
        for cand in [joined.clone(), joined.chars().filter(|c| c.is_alphanumeric()).collect()] {
            self.match_into(&cand, "<joined>", &mut hits);
        }
        hits.sort_by(|a, b| a.prov_id.cmp(&b.prov_id));
        hits.dedup_by(|a, b| a.prov_id == b.prov_id);
        hits
    }

    fn match_into(&self, hay: &str, path: &str, hits: &mut Vec<EgressFinding>) {
        for (value, prov_id, label) in &self.tags {
            if value.len() >= 3 && hay.contains(value.as_str()) {
                let via = if path == "<joined>" { "split" } else { "match" };
                hits.push(EgressFinding {
                    prov_id: prov_id.clone(),
                    label: label.clone(),
                    arg_path: path.to_string(),
                    via: via.to_string(),
                });
            }
        }
    }
}

/// Detector wrapper so taint plugs into the same scoring loop as the content baselines.
pub struct TaintDetector {
    pub store: TaintStore,
}
impl Detector for TaintDetector {
    fn name(&self) -> &str {
        "taint"
    }
    fn scan(&self, arg_strings: &[(String, String)]) -> Vec<EgressFinding> {
        self.store.scan_call(arg_strings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    fn store() -> TaintStore {
        let mut s = TaintStore::new();
        s.tag("123-45-6789", "p-ssn", "ssn");
        s.tag("Maria Alvarez", "p-name", "name");
        s
    }
    #[test]
    fn catches_verbatim() {
        let hits = store().scan_call(&[("body".into(), "patient Maria Alvarez here".into())]);
        assert!(hits.iter().any(|h| h.prov_id == "p-name"));
    }
    #[test]
    fn catches_base64_encoded() {
        let enc = base64::engine::general_purpose::STANDARD.encode("123-45-6789");
        let hits = store().scan_call(&[("body".into(), format!("ref {enc}"))]);
        assert!(hits.iter().any(|h| h.prov_id == "p-ssn"), "taint catches encoded leak");
    }
    #[test]
    fn catches_split_across_fields() {
        let args: Vec<(String, String)> =
            "123-45-6789".chars().enumerate().map(|(i, c)| (format!("f{i}"), c.to_string())).collect();
        let hits = store().scan_call(&args);
        assert!(hits.iter().any(|h| h.prov_id == "p-ssn"), "taint catches field-split leak");
    }
    #[test]
    fn no_false_positive_on_benign() {
        let hits = store().scan_call(&[("body".into(), "refill standing order bed 4".into())]);
        assert!(hits.is_empty());
    }
}
