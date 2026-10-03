//! Loading and indexing the YAML rule library.

use crate::rule::{CompiledRule, Rule};
use crate::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;

/// Load one or more rules from a single YAML file. The file may contain a single
/// rule (a mapping), a list of rules (a sequence), or multiple `---`-separated
/// documents (each a rule or a list of rules).
pub fn load_rules_from_file(path: &Path) -> Result<Vec<Rule>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for doc in serde_yaml::Deserializer::from_str(&text) {
        let value = serde_yaml::Value::deserialize(doc)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        match value {
            serde_yaml::Value::Sequence(items) => {
                for item in items {
                    let rule: Rule = serde_yaml::from_value(item)
                        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
                    out.push(rule);
                }
            }
            serde_yaml::Value::Null => {}
            other => {
                let rule: Rule = serde_yaml::from_value(other)
                    .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
                out.push(rule);
            }
        }
    }
    Ok(out)
}

/// An in-memory index of rules loaded from a directory tree.
pub struct RuleSet {
    rules: HashMap<String, Rule>,
}

impl RuleSet {
    /// Load every `.yaml`/`.yml` rule under `dir` (recursively).
    pub fn load_dir(dir: &Path) -> Result<RuleSet> {
        let mut rules = HashMap::new();
        if !dir.exists() {
            return Err(Error::Config(format!("rules directory not found: {}", dir.display())));
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
            // Skip chain definitions, which live under a `chains` subtree.
            if path.components().any(|c| c.as_os_str() == "chains") {
                continue;
            }
            for rule in load_rules_from_file(path)? {
                if let Some(existing) = rules.insert(rule.id.clone(), rule) {
                    return Err(Error::Config(format!("duplicate rule id: {}", existing.id)));
                }
            }
        }
        Ok(RuleSet { rules })
    }

    /// Construct from an explicit list (used in tests and bench).
    pub fn from_rules(list: Vec<Rule>) -> Result<RuleSet> {
        let mut rules = HashMap::new();
        for r in list {
            if let Some(e) = rules.insert(r.id.clone(), r) {
                return Err(Error::Config(format!("duplicate rule id: {}", e.id)));
            }
        }
        Ok(RuleSet { rules })
    }

    pub fn get(&self, id: &str) -> Result<&Rule> {
        self.rules
            .get(id)
            .ok_or_else(|| Error::Reference(format!("no rule named '{id}'")))
    }

    pub fn contains(&self, id: &str) -> bool {
        self.rules.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// All rules, sorted by id.
    pub fn all(&self) -> Vec<&Rule> {
        let mut v: Vec<&Rule> = self.rules.values().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    /// All rule ids, sorted.
    pub fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.rules.keys().cloned().collect();
        v.sort();
        v
    }

    /// Compile a subset of rules by id into runnable form.
    pub fn compile_subset(&self, ids: &[String]) -> Result<HashMap<String, CompiledRule>> {
        let mut out = HashMap::new();
        for id in ids {
            let rule = self.get(id)?;
            out.insert(id.clone(), rule.compile()?);
        }
        Ok(out)
    }
}
