//! The declarative firewall rule model.
//!
//! A rule is a YAML object declaring an allowed **scope** in natural language
//! and a typed **detector pipeline**. Each node contributes a verdict; the rule
//! collapses its nodes into a per-rule ALLOW/BLOCK/ABSTAIN according to its
//! `short_circuit` policy. The in-scope/out-of-scope exemplars double as the
//! rule's own unit-test fixtures (`qfire rules test`).

use crate::detector::{build_detector, Detector, NodeConfig};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Canonical in-scope / out-of-scope exemplar prompts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Exemplars {
    #[serde(default)]
    pub in_scope: Vec<String>,
    #[serde(default)]
    pub out_of_scope: Vec<String>,
}

/// Labeled fixtures used by `qfire rules test`. Defaults to the exemplars when
/// omitted.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Fixtures {
    #[serde(default)]
    pub in_scope: Vec<String>,
    #[serde(default)]
    pub out_of_scope: Vec<String>,
}

/// How a rule collapses its node verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShortCircuit {
    /// Stop at the first BLOCK (default). Cheap nodes run first to short-circuit
    /// before an expensive llm-judge.
    #[default]
    StopOnFirstBlock,
    /// Stop at the first ALLOW.
    StopOnFirstAllow,
    /// Run every node and aggregate by confidence-weighted majority.
    Aggregate,
}

/// A firewall rule as authored in YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    #[serde(default)]
    pub domain: Option<String>,
    pub scope: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub exemplars: Exemplars,
    #[serde(default)]
    pub short_circuit: ShortCircuit,
    pub pipeline: Vec<NodeConfig>,
    #[serde(default)]
    pub fixtures: Fixtures,
    /// Optional semantic version of the rule, embedded in audit + manifests.
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "1".into()
}

impl Rule {
    /// Load a rule from a YAML file.
    pub fn from_path(path: &Path) -> Result<Rule> {
        let text = std::fs::read_to_string(path)?;
        let rule: Rule = serde_yaml::from_str(&text)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        Ok(rule)
    }

    /// Parse a rule from a YAML string.
    pub fn from_yaml(text: &str) -> Result<Rule> {
        Ok(serde_yaml::from_str(text)?)
    }

    /// The effective test fixtures (explicit fixtures, else the exemplars).
    pub fn effective_fixtures(&self) -> Fixtures {
        if self.fixtures.in_scope.is_empty() && self.fixtures.out_of_scope.is_empty() {
            Fixtures {
                in_scope: self.exemplars.in_scope.clone(),
                out_of_scope: self.exemplars.out_of_scope.clone(),
            }
        } else {
            self.fixtures.clone()
        }
    }

    /// A reproducibility identity: `id@version`.
    pub fn identity(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }

    /// Compile this rule into runnable detectors.
    pub fn compile(&self) -> Result<CompiledRule> {
        if self.pipeline.is_empty() {
            return Err(Error::Config(format!("rule '{}' has an empty pipeline", self.id)));
        }
        let mut detectors = Vec::with_capacity(self.pipeline.len());
        for node in &self.pipeline {
            detectors.push(build_detector(node)?);
        }
        Ok(CompiledRule { rule: self.clone(), detectors })
    }
}

/// A rule with its detector pipeline compiled and ready to evaluate.
pub struct CompiledRule {
    pub rule: Rule,
    pub detectors: Vec<Box<dyn Detector>>,
}

impl CompiledRule {
    pub fn id(&self) -> &str {
        &self.rule.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_rule() {
        let yaml = r#"
id: marketing_scope
scope: "Marketing copy generation only."
pipeline:
  - type: regex
    deny: ['(?i)ignore previous instructions']
  - type: entropy
"#;
        let rule = Rule::from_yaml(yaml).unwrap();
        assert_eq!(rule.id, "marketing_scope");
        assert_eq!(rule.pipeline.len(), 2);
        assert_eq!(rule.short_circuit, ShortCircuit::StopOnFirstBlock);
        rule.compile().unwrap();
    }
}
