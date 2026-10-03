//! Rule chaining and collapse.
//!
//! A **chain** composes many rules into a single terminal decision in one of two
//! interchangeable, individually-benchmarkable modes:
//!
//! - **Ordered** (iptables/kubectl-style): rules evaluated in priority order; the
//!   first BLOCK blocks, the first ALLOW passes, with a configurable default
//!   (default-deny recommended).
//! - **Expression**: a boolean DAG over named rules and groups, e.g.
//!   `injection_guard AND (marketing_scope OR support_scope)`, with AND/OR/NOT,
//!   parallel evaluation of independent branches and short-circuit.

use crate::verdict::Verdict;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Which collapse strategy a chain uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChainMode {
    #[default]
    Ordered,
    Expression,
}

/// Behavior when a detector errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FailPolicy {
    /// Treat detector errors as BLOCK (default; safe).
    #[default]
    FailClosed,
    /// Treat detector errors as pass-through (ABSTAIN).
    FailOpen,
}

/// A chain definition as authored in YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain {
    pub id: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub mode: ChainMode,
    #[serde(default)]
    pub fail_policy: FailPolicy,

    /// Ordered-mode: rules in priority order.
    #[serde(default)]
    pub rules: Vec<String>,
    /// Ordered-mode: terminal default when no rule is decisive.
    #[serde(default = "default_block")]
    pub default: Verdict,

    /// Expression-mode: a boolean expression over rule/group names.
    #[serde(default)]
    pub expression: Option<String>,
    /// Expression-mode: named sub-expressions (groups).
    #[serde(default)]
    pub groups: HashMap<String, String>,

    /// The downstream provider profile to forward to on ALLOW.
    #[serde(default)]
    pub provider: Option<String>,

    /// Apply the de-obfuscation normalization pass (decode Base64/hex/ROT13/URL,
    /// fold homoglyphs/leetspeak, strip zero-width) before detector evaluation.
    /// `true` means "always"; for finer control set `normalize_mode`.
    #[serde(default)]
    pub normalize: bool,

    /// De-obfuscation mode: `off`, `always`, or `triggered` (expand only when the
    /// raw prompt shows an encoding signal). Overrides `normalize` when set.
    #[serde(default)]
    pub normalize_mode: Option<String>,

    #[serde(default = "default_version")]
    pub version: String,
}

fn default_block() -> Verdict {
    Verdict::Block
}
fn default_version() -> String {
    "1".into()
}

impl Chain {
    pub fn from_path(path: &Path) -> Result<Chain> {
        let text = std::fs::read_to_string(path)?;
        let chain: Chain = serde_yaml::from_str(&text)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        chain.validate_shape()?;
        Ok(chain)
    }

    pub fn from_yaml(text: &str) -> Result<Chain> {
        let chain: Chain = serde_yaml::from_str(text)?;
        chain.validate_shape()?;
        Ok(chain)
    }

    fn validate_shape(&self) -> Result<()> {
        match self.mode {
            ChainMode::Ordered => {
                if self.rules.is_empty() {
                    return Err(Error::Config(format!(
                        "ordered chain '{}' has no rules",
                        self.id
                    )));
                }
            }
            ChainMode::Expression => {
                if self.expression.is_none() {
                    return Err(Error::Config(format!(
                        "expression chain '{}' has no expression",
                        self.id
                    )));
                }
                // Parse to validate.
                self.parse_expression()?;
            }
        }
        Ok(())
    }

    /// All rule ids this chain references (deduped), regardless of mode.
    pub fn referenced_rules(&self) -> Result<Vec<String>> {
        match self.mode {
            ChainMode::Ordered => {
                let mut seen = BTreeSet::new();
                let mut out = Vec::new();
                for r in &self.rules {
                    if seen.insert(r.clone()) {
                        out.push(r.clone());
                    }
                }
                Ok(out)
            }
            ChainMode::Expression => {
                let expr = self.parse_expression()?;
                let mut set = BTreeSet::new();
                expr.collect_rules(&mut set);
                Ok(set.into_iter().collect())
            }
        }
    }

    /// Parse the expression, inlining group references.
    pub fn parse_expression(&self) -> Result<Expr> {
        let src = self
            .expression
            .as_ref()
            .ok_or_else(|| Error::Expression("no expression".into()))?;
        parse_with_groups(src, &self.groups, 0)
    }

    pub fn identity(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }
}

/// A boolean expression over rule predicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// A reference to a rule; the predicate is "this rule ALLOWed".
    Rule(String),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Collect all referenced rule names.
    pub fn collect_rules(&self, out: &mut BTreeSet<String>) {
        match self {
            Expr::Rule(r) => {
                out.insert(r.clone());
            }
            Expr::Not(e) => e.collect_rules(out),
            Expr::And(a, b) | Expr::Or(a, b) => {
                a.collect_rules(out);
                b.collect_rules(out);
            }
        }
    }

    /// Evaluate the expression given a map of rule → did-it-ALLOW.
    /// Returns (overall, decisive_rule) where decisive_rule names a rule whose
    /// value determined a falsey result (for explaining BLOCKs).
    pub fn eval(&self, vals: &HashMap<String, bool>) -> bool {
        match self {
            Expr::Rule(r) => *vals.get(r).unwrap_or(&false),
            Expr::Not(e) => !e.eval(vals),
            Expr::And(a, b) => a.eval(vals) && b.eval(vals),
            Expr::Or(a, b) => a.eval(vals) || b.eval(vals),
        }
    }
}

// ---- Expression parser (recursive descent) -------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    And,
    Or,
    Not,
    LParen,
    RParen,
}

fn lex(src: &str) -> Result<Vec<Tok>> {
    let mut toks = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '(' {
            chars.next();
            toks.push(Tok::LParen);
        } else if c == ')' {
            chars.next();
            toks.push(Tok::RParen);
        } else if c == '!' {
            chars.next();
            toks.push(Tok::Not);
        } else if c == '&' {
            chars.next();
            if chars.peek() == Some(&'&') {
                chars.next();
            }
            toks.push(Tok::And);
        } else if c == '|' {
            chars.next();
            if chars.peek() == Some(&'|') {
                chars.next();
            }
            toks.push(Tok::Or);
        } else if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
            let mut ident = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
                    ident.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            match ident.to_ascii_uppercase().as_str() {
                "AND" => toks.push(Tok::And),
                "OR" => toks.push(Tok::Or),
                "NOT" => toks.push(Tok::Not),
                _ => toks.push(Tok::Ident(ident)),
            }
        } else {
            return Err(Error::Expression(format!("unexpected character '{c}'")));
        }
    }
    Ok(toks)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    groups: HashMap<String, String>,
    depth: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Tok::Or)) {
            self.next();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), Some(Tok::And)) {
            self.next();
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr> {
        if matches!(self.peek(), Some(Tok::Not)) {
            self.next();
            let e = self.parse_not()?;
            Ok(Expr::Not(Box::new(e)))
        } else {
            self.parse_atom()
        }
    }

    fn parse_atom(&mut self) -> Result<Expr> {
        match self.next() {
            Some(Tok::LParen) => {
                let e = self.parse_or()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err(Error::Expression("expected ')'".into())),
                }
            }
            Some(Tok::Ident(name)) => {
                // Group reference? Inline it (with recursion-depth guard).
                if let Some(sub) = self.groups.get(&name).cloned() {
                    if self.depth > 32 {
                        return Err(Error::Expression(format!(
                            "group nesting too deep at '{name}' (cycle?)"
                        )));
                    }
                    parse_with_groups(&sub, &self.groups, self.depth + 1)
                } else {
                    Ok(Expr::Rule(name))
                }
            }
            other => Err(Error::Expression(format!("unexpected token {other:?}"))),
        }
    }
}

fn parse_with_groups(src: &str, groups: &HashMap<String, String>, depth: usize) -> Result<Expr> {
    let toks = lex(src)?;
    if toks.is_empty() {
        return Err(Error::Expression("empty expression".into()));
    }
    let mut parser = Parser { toks, pos: 0, groups: groups.clone(), depth };
    let expr = parser.parse_or()?;
    if parser.pos != parser.toks.len() {
        return Err(Error::Expression("trailing tokens in expression".into()));
    }
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_or_not() {
        let e = parse_with_groups("a AND (b OR c) AND NOT d", &HashMap::new(), 0).unwrap();
        let mut set = BTreeSet::new();
        e.collect_rules(&mut set);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn evaluates_expression() {
        let e = parse_with_groups("a AND (b OR c)", &HashMap::new(), 0).unwrap();
        let mut vals = HashMap::new();
        vals.insert("a".into(), true);
        vals.insert("b".into(), false);
        vals.insert("c".into(), true);
        assert!(e.eval(&vals));
        vals.insert("a".into(), false);
        assert!(!e.eval(&vals));
    }

    #[test]
    fn inlines_groups() {
        let mut groups = HashMap::new();
        groups.insert("scopes".to_string(), "marketing OR support".to_string());
        let e = parse_with_groups("guard AND scopes", &groups, 0).unwrap();
        let mut set = BTreeSet::new();
        e.collect_rules(&mut set);
        assert!(set.contains("marketing"));
        assert!(set.contains("support"));
        assert!(set.contains("guard"));
    }

    #[test]
    fn supports_symbolic_operators() {
        let e = parse_with_groups("a && (b || c) && !d", &HashMap::new(), 0).unwrap();
        let mut vals = HashMap::new();
        vals.insert("a".into(), true);
        vals.insert("b".into(), true);
        vals.insert("c".into(), false);
        vals.insert("d".into(), false);
        assert!(e.eval(&vals));
    }
}
